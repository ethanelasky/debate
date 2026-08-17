'use strict';

const http = require('http');

function getJson(path) {
    return new Promise((resolve, reject) => {
        const request = http.get(
            {
                host: '127.0.0.1',
                port: 2000,
                path,
                timeout: 3000,
                headers: { Accept: 'application/json' },
            },
            response => {
                let body = '';
                response.setEncoding('utf8');
                response.on('data', chunk => {
                    body += chunk;
                    if (body.length > 1024 * 1024) {
                        request.destroy(new Error(`${path} response is too large`));
                    }
                });
                response.on('end', () => {
                    if (response.statusCode !== 200) {
                        return reject(
                            new Error(`${path} returned HTTP ${response.statusCode}`)
                        );
                    }
                    try {
                        resolve(JSON.parse(body));
                    } catch (error) {
                        reject(new Error(`${path} returned invalid JSON`));
                    }
                });
            }
        );
        request.on('timeout', () => request.destroy(new Error(`${path} timed out`)));
        request.on('error', reject);
    });
}

(async () => {
    const root = await getJson('/');
    if (
        !root ||
        typeof root.message !== 'string' ||
        !/^Piston v[^\s]+$/.test(root.message)
    ) {
        throw new Error('GET / did not identify Piston');
    }

    const runtimes = await getJson('/api/v2/runtimes');
    if (
        !Array.isArray(runtimes) ||
        !runtimes.some(
            runtime =>
                runtime &&
                runtime.language === 'python' &&
                runtime.version === '3.12.0'
        )
    ) {
        throw new Error('exact python-3.12.0 runtime is not ready');
    }
})().catch(error => {
    console.error(error.message);
    process.exit(1);
});
