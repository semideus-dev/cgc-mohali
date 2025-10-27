const env = {
  databaseUrl: process.env.DATABASE_URL,
  appUrl: process.env.BETTER_AUTH_URL || "http://localhost:3000",
  s3Token: process.env.UPLOADTHING_TOKEN,
};

export default env;
