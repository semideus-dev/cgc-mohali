"""Comprehensive schema update - Add all tables and enhanced fields

Revision ID: 002_comprehensive
Revises: 001_initial
Create Date: 2025-10-27 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision: str = '002_comprehensive'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add all new tables and enhance existing ones"""
    
    # Create user table
    op.create_table(
        'user',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('email', sa.Text(), nullable=False),
        sa.Column('email_verified', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('image', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_user_email', 'user', ['email'], unique=True)
    
    # Create session table
    op.create_table(
        'session',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('token', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('ip_address', sa.Text(), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('user_id', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_session_token', 'session', ['token'], unique=True)
    
    # Create account table
    op.create_table(
        'account',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('account_id', sa.Text(), nullable=False),
        sa.Column('provider_id', sa.Text(), nullable=False),
        sa.Column('user_id', sa.Text(), nullable=False),
        sa.Column('access_token', sa.Text(), nullable=True),
        sa.Column('refresh_token', sa.Text(), nullable=True),
        sa.Column('id_token', sa.Text(), nullable=True),
        sa.Column('access_token_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('refresh_token_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('scope', sa.Text(), nullable=True),
        sa.Column('password', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create verification table
    op.create_table(
        'verification',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('identifier', sa.Text(), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create canvas table
    op.create_table(
        'canvas',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('url', sa.Text(), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('user_id', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Enhance analysis_jobs table
    op.add_column('analysis_jobs', sa.Column('user_id', sa.Text(), nullable=True))
    op.add_column('analysis_jobs', sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()))
    op.add_column('analysis_jobs', sa.Column('user_prompt', sa.Text(), nullable=True))
    op.add_column('analysis_jobs', sa.Column('ocr_results', JSONB, nullable=True))
    op.add_column('analysis_jobs', sa.Column('text_analysis', JSONB, nullable=True))
    op.add_column('analysis_jobs', sa.Column('image_analysis', JSONB, nullable=True))
    op.add_column('analysis_jobs', sa.Column('critique', JSONB, nullable=True))
    op.add_column('analysis_jobs', sa.Column('master_prompt', sa.Text(), nullable=True))
    op.add_column('analysis_jobs', sa.Column('generated_image_url', sa.String(), nullable=True))
    
    # Add foreign key
    op.create_foreign_key(
        'fk_analysis_jobs_user',
        'analysis_jobs', 'user',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # Update status column default
    op.alter_column('analysis_jobs', 'status', server_default='pending')


def downgrade() -> None:
    """Revert all changes"""
    
    # Drop foreign key and columns from analysis_jobs
    op.drop_constraint('fk_analysis_jobs_user', 'analysis_jobs', type_='foreignkey')
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
    
    # Drop new tables
    op.drop_table('canvas')
    op.drop_table('verification')
    op.drop_table('account')
    op.drop_index('ix_session_token', table_name='session')
    op.drop_table('session')
    op.drop_index('ix_user_email', table_name='user')
    op.drop_table('user')

