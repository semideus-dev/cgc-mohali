"""Initial migration - Create analysis_jobs table

Revision ID: 001_initial
Revises: 
Create Date: 2025-10-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the analysis_jobs table"""
    op.create_table(
        'analysis_jobs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='processing'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('original_image_url', sa.String(), nullable=True),
        sa.Column('results', JSONB, nullable=True),
    )
    
    # Create indexes
    op.create_index('ix_analysis_jobs_id', 'analysis_jobs', ['id'])
    op.create_index('ix_analysis_jobs_status', 'analysis_jobs', ['status'])


def downgrade() -> None:
    """Drop the analysis_jobs table"""
    op.drop_index('ix_analysis_jobs_status', table_name='analysis_jobs')
    op.drop_index('ix_analysis_jobs_id', table_name='analysis_jobs')
    op.drop_table('analysis_jobs')

