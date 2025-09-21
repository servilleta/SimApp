// Lightweight logger that no-ops in production
const isProd = import.meta.env.PROD;

const noop = () => {};

const logger = {
  debug: isProd ? noop : (...args) => console.debug(...args),
  info: isProd ? noop : (...args) => console.info(...args),
  warn: isProd ? noop : (...args) => console.warn(...args),
  error: isProd ? noop : (...args) => console.error(...args),
};

export default logger;




