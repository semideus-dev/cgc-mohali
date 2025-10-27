"""Add enhanced fields to analysis_jobs table

Revision ID: 003_add_enhanced_fields
Revises: 001_initial
Create Date: 2025-10-27 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = '003_add_enhanced_fields'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add enhanced fields to analysis_jobs table (safe version)"""
    
    # Get connection to check existing columns
    conn = op.get_bind()
    
    # Check which columns already exist
    result = conn.execute(sa.text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'analysis_jobs'
    """))
    existing_columns = {row[0] for row in result}
    
    # Define columns to add
    columns_to_add = [
        ('user_id', sa.Column('user_id', sa.Text(), nullable=True)),
        ('updated_at', sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)),
        ('user_prompt', sa.Column('user_prompt', sa.Text(), nullable=True)),
        ('ocr_results', sa.Column('ocr_results', JSONB, nullable=True)),
        ('text_analysis', sa.Column('text_analysis', JSONB, nullable=True)),
        ('image_analysis', sa.Column('image_analysis', JSONB, nullable=True)),
        ('critique', sa.Column('critique', JSONB, nullable=True)),
        ('master_prompt', sa.Column('master_prompt', sa.Text(), nullable=True)),
        ('generated_image_url', sa.Column('generated_image_url', sa.String(), nullable=True))
    ]
    
    # Add only columns that don't exist
    for col_name, col_def in columns_to_add:
        if col_name not in existing_columns:
            print(f"Adding column: {col_name}")
            op.add_column('analysis_jobs', col_def)
        else:
            print(f"Column {col_name} already exists, skipping")
    
    # Update status default
    try:
        op.alter_column('analysis_jobs', 'status', server_default='pending')
    except Exception as e:
        print(f"Could not update status default: {e}")
    
    # Update existing records to have updated_at if the column was added
    if 'updated_at' not in existing_columns:
        op.execute("UPDATE analysis_jobs SET updated_at = created_at WHERE updated_at IS NULL")


def downgrade() -> None:
    """Remove enhanced fields from analysis_jobs table"""
    
    # Remove added columns
    op.drop_column('analysis_jobs', 'generated_image_url')
    op.drop_column('analysis_jobs', 'master_prompt')
    op.drop_column('analysis_jobs', 'critique')
    op.drop_column('analysis_jobs', 'image_analysis')
    op.drop_column('analysis_jobs', 'text_analysis')
    op.drop_column('analysis_jobs', 'ocr_results')
    op.drop_column('analysis_jobs', 'user_prompt')
    op.drop_column('analysis_jobs', 'updated_at')
    op.drop_column('analysis_jobs', 'user_id')
    
    # Revert status default
    op.alter_column('analysis_jobs', 'status', server_default='processing')
