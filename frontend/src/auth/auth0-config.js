export const auth0Config = {
    domain: "dev-jw6k27f0v5tcgl56.eu.auth0.com",
    clientId: "UDXorRodTlUmgkigfaWW81Rr40gKpeAJ",
    audience: "https://simapp.ai/api",
    redirectUri: "http://localhost:8000/callback",
    logoutRedirectUri: "http://localhost:8000",
    scope: "openid profile email offline_access",
    responseType: "code",
    responseMode: "query"
}; 