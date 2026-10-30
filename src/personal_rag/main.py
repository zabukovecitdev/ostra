import logging

from personal_rag.database import ensure_database


def main():
    logging.basicConfig(level=logging.INFO)
    database_path = ensure_database()
    print(f"Database ready at {database_path}")


if __name__ == "__main__":
    main()
