/**
 * Test EMAIL-OTP login (src/dashboard/email-otp-login.js).
 * Uses injected deps (fake transport + fake code reader) — zero real network calls.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import {
  sendEmailVerification,
  registerUserWithOtp,
  pollGmailForOtp,
  windsurfLoginViaEmailOtp,
  __setOtpTransportForTests,
} from '../src/dashboard/email-otp-login.js';

describe('email-otp-login', () => {
  describe('sendEmailVerification', () => {
    it('builds correct Connect-RPC body and parses 200 response', async () => {
      const capturedCalls = [];
      const fakeTransport = (url, opts, postData) => {
        capturedCalls.push({ url, opts, body: postData });
        if (url.includes('SendEmailVerification')) {
          return Promise.resolve({ status: 200, data: {} });
        }
        return Promise.reject(new Error('Unknown URL'));
      };
      __setOtpTransportForTests(fakeTransport);

      const result = await sendEmailVerification(
        'test@example.com',
        'Test User',
        'fake-turnstile-token',
        null
      );

      assert.equal(result.success, true);
      assert.equal(capturedCalls.length, 1);
      const call = capturedCalls[0];
      assert.ok(call.url.includes('SendEmailVerification'));
      assert.equal(call.opts.method, 'POST');
      assert.equal(call.opts.headers['Connect-Protocol-Version'], '1');

      const bodyObj = JSON.parse(call.body);
      assert.equal(bodyObj.email, 'test@example.com');
      assert.equal(bodyObj.first_name, 'Test User');
      assert.equal(bodyObj.turnstile_token, 'fake-turnstile-token');

      __setOtpTransportForTests(null); // reset
    });

    it('throws ERR_TURNSTILE_INVALID on 400 with turnstile error', async () => {
      const fakeTransport = () => {
        return Promise.resolve({
          status: 400,
          data: 'turnstile validation failed',
        });
      };
      __setOtpTransportForTests(fakeTransport);

      await assert.rejects(
        () => sendEmailVerification('test@example.com', 'Test', 'bad-token', null),
        /ERR_TURNSTILE_INVALID/
      );

      __setOtpTransportForTests(null);
    });
  });

  describe('registerUserWithOtp', () => {
    it('parses api_key from RegisterUserResponse', async () => {
      const fakeTransport = (url) => {
        if (url.includes('RegisterUser')) {
          return Promise.resolve({
            status: 200,
            data: {
              api_key: 'sk-ws-01-fake',
              name: 'Test User',
              api_server_url: 'https://server.windsurf.com',
            },
          });
        }
        return Promise.reject(new Error('Unknown URL'));
      };
      __setOtpTransportForTests(fakeTransport);

      const result = await registerUserWithOtp(
        'test@example.com',
        '123456',
        'fake-turnstile-token',
        'Test',
        'User',
        null
      );

      assert.equal(result.apiKey, 'sk-ws-01-fake');
      assert.equal(result.name, 'Test User');
      assert.equal(result.apiServerUrl, 'https://server.windsurf.com');

      __setOtpTransportForTests(null);
    });

    it('throws ERR_OTP_INVALID on 400 with otp error', async () => {
      const fakeTransport = () => {
        return Promise.resolve({
          status: 400,
          data: 'invalid otp code',
        });
      };
      __setOtpTransportForTests(fakeTransport);

      await assert.rejects(
        () => registerUserWithOtp('test@example.com', '999999', 'fake-token', '', '', null),
        /ERR_OTP_INVALID/
      );

      __setOtpTransportForTests(null);
    });
  });

  describe('pollGmailForOtp', () => {
    it('throws ERR_GMAIL_CREDS_MISSING when creds missing', async () => {
      await assert.rejects(
        () => pollGmailForOtp('', '', 1000),
        /ERR_GMAIL_CREDS_MISSING/
      );
    });
  });

  describe('windsurfLoginViaEmailOtp (orchestrator)', () => {
    it('wires sendEmail → readOtp → register with injected deps', async () => {
      const email = 'test@example.com';
      const firstName = 'Test';
      const turnstileToken = 'fake-turnstile-token';
      const fakeOtpCode = '123456';

      const fakeCalls = [];
      const fakeTransport = (url, opts, postData) => {
        fakeCalls.push({ url, opts, body: postData });
        if (url.includes('SendEmailVerification')) {
          return Promise.resolve({ status: 200, data: {} });
        }
        if (url.includes('RegisterUser')) {
          return Promise.resolve({
            status: 200,
            data: { api_key: 'sk-ws-01-fake', name: 'Test User', api_server_url: '' },
          });
        }
        return Promise.reject(new Error('Unknown URL'));
      };
      __setOtpTransportForTests(fakeTransport);

      // Fake OTP reader (no real IMAP)
      const fakeReadOtp = async () => fakeOtpCode;

      // Fake env
      const fakeEnv = {
        GMAIL_IMAP_USER: 'test@gmail.com',
        GMAIL_IMAP_PASSWORD: 'app-password',
      };

      const result = await windsurfLoginViaEmailOtp(email, firstName, turnstileToken, '', null, {
        readOtp: fakeReadOtp,
        env: fakeEnv,
      });

      assert.equal(result.apiKey, 'sk-ws-01-fake');
      assert.equal(result.email, email);
      assert.equal(fakeCalls.length, 2);
      assert.ok(fakeCalls[0].url.includes('SendEmailVerification'));
      assert.ok(fakeCalls[1].url.includes('RegisterUser'));

      // Verify RegisterUser body includes otp_code
      const registerBody = JSON.parse(fakeCalls[1].body);
      assert.equal(registerBody.otp_code, fakeOtpCode);
      assert.equal(registerBody.email, email);

      __setOtpTransportForTests(null);
    });

    it('throws ERR_GMAIL_CREDS_MISSING when env vars not set', async () => {
      const fakeEnv = {}; // no GMAIL_IMAP_USER / GMAIL_IMAP_PASSWORD

      // Need fake transport so sendEmailVerification doesn't fail first
      const fakeTransport = (url) => {
        if (url.includes('SendEmailVerification')) {
          return Promise.resolve({ status: 200, data: {} });
        }
        return Promise.reject(new Error('Unknown URL'));
      };
      __setOtpTransportForTests(fakeTransport);

      await assert.rejects(
        () => windsurfLoginViaEmailOtp(
          'test@example.com',
          'Test',
          'fake-turnstile-token',
          '',
          null,
          { env: fakeEnv }
        ),
        /ERR_GMAIL_CREDS_MISSING/
      );

      __setOtpTransportForTests(null);
    });
  });

  describe('Wire protocol compliance', () => {
    it('sendEmailVerification builds Connect-RPC JSON with snake_case fields', async () => {
      const capturedCalls = [];
      const fakeTransport = (url, opts, postData) => {
        capturedCalls.push({ url, opts, body: postData });
        return Promise.resolve({ status: 200, data: {} });
      };
      __setOtpTransportForTests(fakeTransport);

      await sendEmailVerification('test@example.com', 'John', 'token123', null);

      const bodyObj = JSON.parse(capturedCalls[0].body);
      assert.ok('first_name' in bodyObj); // snake_case
      assert.ok('email' in bodyObj);
      assert.ok('turnstile_token' in bodyObj);
      assert.ok(!('firstName' in bodyObj)); // NOT camelCase

      __setOtpTransportForTests(null);
    });

    it('registerUserWithOtp parses both api_key and apiKey variants', async () => {
      // Test snake_case response
      const fakeTransport1 = () => Promise.resolve({
        status: 200,
        data: { api_key: 'sk-1', name: 'User', api_server_url: '' },
      });
      __setOtpTransportForTests(fakeTransport1);

      const result1 = await registerUserWithOtp('t@e.com', '123456', 'tok', '', '', null);
      assert.equal(result1.apiKey, 'sk-1');

      // Test camelCase response (future-proofing)
      const fakeTransport2 = () => Promise.resolve({
        status: 200,
        data: { apiKey: 'sk-2', name: 'User', apiServerUrl: '' },
      });
      __setOtpTransportForTests(fakeTransport2);

      const result2 = await registerUserWithOtp('t@e.com', '123456', 'tok', '', '', null);
      assert.equal(result2.apiKey, 'sk-2');

      __setOtpTransportForTests(null);
    });
  });
});
