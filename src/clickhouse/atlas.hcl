env {
  name = atlas.env
  url  = getenv("DATABASE_URL")
  src = "file://schema.sql" 
  migration {
    dir = "file://migrations"
  }
}