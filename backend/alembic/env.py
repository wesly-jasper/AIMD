import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, text
from alembic import context

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import settings
from app.db.models import Base

config = context.config

# Only the alembic CLI configures logging from alembic.ini. When the application
# runs migrations at startup (app.db.database.init_db) there is no config file,
# and calling fileConfig would replace the application's logging setup.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    # An explicitly supplied URL (tests migrating a scratch database) wins over
    # the application setting.
    return config.get_main_option("sqlalchemy.url") or settings.database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        is_sqlite = connection.dialect.name == "sqlite"

        if is_sqlite:
            # SQLite cannot drop columns or add constraints in place, so batch
            # migrations rebuild the table and DROP the original. With foreign
            # keys enforced, that DROP fires ON DELETE CASCADE into every child
            # table. The pragma is a no-op inside a transaction, so it is issued
            # here, before alembic opens one.
            connection.execute(text("PRAGMA foreign_keys=OFF"))
            connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()

        if is_sqlite:
            # Enforcement was off, so prove nothing was orphaned rather than
            # assume it.
            violations = connection.execute(text("PRAGMA foreign_key_check")).fetchall()
            if violations:
                raise RuntimeError(
                    f"Migration left {len(violations)} foreign-key violation(s); "
                    f"first: {tuple(violations[0])}"
                )


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
