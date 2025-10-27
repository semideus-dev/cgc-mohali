const env = {
  databaseUrl: process.env.DATABASE_URL,
  appUrl: process.env.BETTER_AUTH_URL || "http://localhost:3000",
};

export default env;
