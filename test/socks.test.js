import { describe, it, before, after, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import net from 'node:net';
import { isSocks, createSocksTunnel } from '../src/socks.js';
import { config } from '../src/config.js';

// createSocksTunnel dials the proxy host through resolveProxyConnectHost, which
// rejects private/loopback IPs unless allowPrivateProxyHosts is set. Our fake
// SOCKS server binds 127.0.0.1, so flip the flag for the duration of the suite.
let savedAllow;
before(() => { savedAllow = config.allowPrivateProxyHosts; config.allowPrivateProxyHosts = true; });
after(() => { config.allowPrivateProxyHosts = savedAllow; });

const SOCKS_VERSION = 0x05;

// A minimal scriptable SOCKS5 server. `behavior` tunes each phase so we can
// exercise the happy path, auth, and every error branch of the client.
function startFakeSocksServer(behavior = {}) {
  const {
    greetingVersion = SOCKS_VERSION,
    offerMethod = null,        // null → echo what makes sense; else force a method byte
    requireUserPass = false,
    authStatus = 0x00,
    connectRep = 0x00,
    connectAtyp = 0x01,        // IPv4 reply
  } = behavior;

  const seen = { greeting: null, auth: null, connect: null };

  const server = net.createServer((sock) => {
    let phase = 'greeting';
    let buf = Buffer.alloc(0);
    sock.on('data', (chunk) => {
      buf = Buffer.concat([buf, chunk]);
      if (phase === 'greeting') {
        if (buf.length < 2) return;
        const nMethods = buf[1];
        if (buf.length < 2 + nMethods) return;
        seen.greeting = Buffer.from(buf.subarray(0, 2 + nMethods));
        const methods = [...buf.subarray(2, 2 + nMethods)];
        buf = buf.subarray(2 + nMethods);
        const method = offerMethod != null
          ? offerMethod
          : (requireUserPass ? 0x02 : (methods.includes(0x00) ? 0x00 : 0xFF));
        sock.write(Buffer.from([greetingVersion, method]));
        phase = method === 0x02 ? 'auth' : 'connect';
      } else if (phase === 'auth') {
        // ver(1) ulen(1) user plen(1) pass — just consume it.
        if (buf.length < 2) return;
        const ulen = buf[1];
        if (buf.length < 2 + ulen + 1) return;
        const plen = buf[2 + ulen];
        if (buf.length < 3 + ulen + plen) return;
        seen.auth = Buffer.from(buf.subarray(0, 3 + ulen + plen));
        buf = buf.subarray(3 + ulen + plen);
        sock.write(Buffer.from([0x01, authStatus]));
        phase = authStatus === 0x00 ? 'connect' : 'done';
      } else if (phase === 'connect') {
        if (buf.length < 4) return;
        seen.connect = Buffer.from(buf);
        buf = Buffer.alloc(0);
        // Reply: ver, rep, rsv, atyp, addr, port
        const addr = connectAtyp === 0x01 ? [0, 0, 0, 0] : [];
        sock.write(Buffer.from([SOCKS_VERSION, connectRep, 0x00, connectAtyp, ...addr, 0x00, 0x00]));
        phase = 'done';
      }
    });
  });

  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      resolve({ server, port: server.address().port, seen });
    });
  });
}

describe('isSocks', () => {
  it('recognizes socks5 / socks / socks5h', () => {
    for (const t of ['socks5', 'socks', 'socks5h', 'SOCKS5']) {
      assert.equal(isSocks({ type: t }), true, t);
    }
  });
  it('rejects http and empty', () => {
    assert.equal(isSocks({ type: 'http' }), false);
    assert.equal(isSocks({}), false);
    assert.equal(isSocks(null), false);
  });
});

describe('createSocksTunnel', () => {
  let fake;
  afterEach(() => { if (fake?.server) fake.server.close(); fake = null; });

  it('completes the no-auth handshake and resolves a connected socket', async () => {
    fake = await startFakeSocksServer();
    const sock = await createSocksTunnel({ host: '127.0.0.1', port: fake.port }, 'example.com', 443);
    assert.ok(sock instanceof net.Socket, 'returns a connected socket');
    // Greeting offered exactly the no-auth method.
    assert.deepEqual([...fake.seen.greeting], [0x05, 0x01, 0x00]);
    // CONNECT used ATYP_DOMAIN with the target name + port 443.
    assert.equal(fake.seen.connect[3], 0x03, 'ATYP domain');
    const nameLen = fake.seen.connect[4];
    assert.equal(fake.seen.connect.subarray(5, 5 + nameLen).toString(), 'example.com');
    assert.equal(fake.seen.connect.readUInt16BE(5 + nameLen), 443);
    sock.destroy();
  });

  it('performs username/password auth (RFC 1929) when credentials are given', async () => {
    fake = await startFakeSocksServer({ requireUserPass: true });
    const sock = await createSocksTunnel(
      { host: '127.0.0.1', port: fake.port, username: 'alice', password: 'secret' },
      'host.test', 8080,
    );
    assert.ok(sock instanceof net.Socket);
    // Greeting offered BOTH no-auth and userpass.
    assert.deepEqual([...fake.seen.greeting], [0x05, 0x02, 0x00, 0x02]);
    // Auth frame carried the credentials.
    assert.ok(fake.seen.auth, 'auth frame received');
    const ulen = fake.seen.auth[1];
    assert.equal(fake.seen.auth.subarray(2, 2 + ulen).toString(), 'alice');
    sock.destroy();
  });

  it('rejects when the server offers no acceptable auth method (0xFF)', async () => {
    fake = await startFakeSocksServer({ offerMethod: 0xFF });
    await assert.rejects(
      createSocksTunnel({ host: '127.0.0.1', port: fake.port }, 'example.com', 443),
      /no acceptable auth method/,
    );
  });

  it('rejects on a bad SOCKS version in the greeting reply', async () => {
    fake = await startFakeSocksServer({ greetingVersion: 0x04 });
    await assert.rejects(
      createSocksTunnel({ host: '127.0.0.1', port: fake.port }, 'example.com', 443),
      /server version 4 unsupported/,
    );
  });

  it('rejects when username/password auth fails (status != 0)', async () => {
    fake = await startFakeSocksServer({ requireUserPass: true, authStatus: 0x01 });
    await assert.rejects(
      createSocksTunnel(
        { host: '127.0.0.1', port: fake.port, username: 'x', password: 'y' },
        'example.com', 443,
      ),
      /authentication failed/,
    );
  });

  it('maps a non-zero CONNECT reply code to a human reason', async () => {
    fake = await startFakeSocksServer({ connectRep: 0x05 }); // connection refused
    await assert.rejects(
      createSocksTunnel({ host: '127.0.0.1', port: fake.port }, 'example.com', 443),
      /connection refused/,
    );
  });

  it('honors the timeout against a silent server', async () => {
    // Server that accepts the socket but never replies to the greeting.
    const silent = net.createServer(() => { /* swallow, never respond */ });
    await new Promise((r) => silent.listen(0, '127.0.0.1', r));
    const port = silent.address().port;
    try {
      await assert.rejects(
        createSocksTunnel({ host: '127.0.0.1', port }, 'example.com', 443, 150),
        /timeout/,
      );
    } finally {
      silent.close();
    }
  });
});
