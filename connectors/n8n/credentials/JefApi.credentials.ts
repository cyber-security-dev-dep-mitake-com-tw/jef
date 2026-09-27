import type {
  IAuthenticateGeneric,
  ICredentialTestRequest,
  ICredentialType,
  INodeProperties,
} from 'n8n-workflow';

export class JefApi implements ICredentialType {
  name = 'jefApi';

  displayName = 'JEF API';

  documentationUrl = 'https://github.com/cyber-security-dev-dep-mitake-com-tw/jef';

  properties: INodeProperties[] = [
    {
      displayName: 'Base URL',
      name: 'baseUrl',
      type: 'string',
      default: 'http://localhost:8080',
      required: true,
      placeholder: 'http://jef.internal:8080',
      description: 'Base URL of the JEF server',
    },
    {
      displayName: 'API Key',
      name: 'apiKey',
      type: 'string',
      typeOptions: { password: true },
      default: '',
      description:
        'Optional bearer token. JEF itself does not authenticate; set this when it sits behind an authenticating proxy.',
    },
  ];

  authenticate: IAuthenticateGeneric = {
    type: 'generic',
    properties: {
      headers: {
        // An empty key must not produce "Bearer " -- some proxies treat a
        // malformed header as worse than an absent one.
        Authorization: '={{ $credentials.apiKey ? "Bearer " + $credentials.apiKey : undefined }}',
      },
    },
  };

  test: ICredentialTestRequest = {
    request: {
      baseURL: '={{ $credentials.baseUrl.replace(/\\/+$/, "") }}',
      url: '/healthz',
    },
  };
}
