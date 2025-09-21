export const auth0Config = {
    domain: import.meta.env.VITE_AUTH0_DOMAIN || "dev-jw6k27f0v5tcgl56.eu.auth0.com",
    clientId: import.meta.env.VITE_AUTH0_CLIENT_ID || "UDXorRodTlUmgkigfaWW81Rr40gKpeAJ",
    audience: import.meta.env.VITE_AUTH0_AUDIENCE || "https://simapp.ai/api",
    redirectUri: "http://localhost:9090/callback",
    logoutRedirectUri: "http://localhost:9090",
    scope: "openid profile email offline_access",
    responseType: "code",
    responseMode: "query"
}; 