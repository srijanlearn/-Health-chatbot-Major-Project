"""
Unit tests for tenant management — ID validation, config loading, path resolution.
"""

import pytest
import yaml
from app.tenant import validate_tenant_id, TenantConfig, TenantManager, DEFAULT_TENANT_ID


# ── ID Validation ──────────────────────────────────────────────────────────────

class TestValidateTenantId:
    @pytest.mark.parametrize("tid", [
        "default",
        "apollo_delhi",
        "MedPlus-123",
        "a",
        "A" * 64,
        "hospital-01",
    ])
    def test_valid_ids(self, tid):
        assert validate_tenant_id(tid) is True

    @pytest.mark.parametrize("tid", [
        "",
        "../etc/passwd",
        "a" * 65,
        "has space",
        "has/slash",
        "has.dot",
        "has@symbol",
        "null\x00byte",
    ])
    def test_invalid_ids(self, tid):
        assert validate_tenant_id(tid) is False


# ── TenantConfig ───────────────────────────────────────────────────────────────

class TestTenantConfig:
    def test_default_tenant_uses_legacy_kg_path(self):
        config = TenantConfig(tenant_id=DEFAULT_TENANT_ID)
        assert config.resolved_kg_db_path == "data/knowledge.db"

    def test_default_tenant_uses_legacy_vector_store(self):
        config = TenantConfig(tenant_id=DEFAULT_TENANT_ID)
        assert config.resolved_vector_store_base == "./db"

    def test_custom_tenant_derives_kg_path_from_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = TenantConfig(tenant_id="hospital_a")
        # Should be tenants/hospital_a/knowledge.db
        assert "hospital_a" in config.resolved_kg_db_path
        assert config.resolved_kg_db_path.endswith("knowledge.db")

    def test_custom_tenant_vector_store_is_scoped(self):
        config = TenantConfig(tenant_id="pharmacy_b")
        assert "pharmacy_b" in config.resolved_vector_store_base

    def test_explicit_kg_path_overrides_default(self):
        config = TenantConfig(tenant_id="any", kg_db_path="/custom/path.db")
        assert config.resolved_kg_db_path == "/custom/path.db"

    def test_explicit_vector_store_overrides_default(self):
        config = TenantConfig(tenant_id="any", vector_store_base="/custom/vectors")
        assert config.resolved_vector_store_base == "/custom/vectors"

    def test_load_from_yaml_file(self, tmp_path):
        tid = "test_tenant"
        config_dir = tmp_path / "tenants" / tid
        config_dir.mkdir(parents=True)
        yaml_data = {
            "name": "Test Hospital",
            "kg_db_path": "/data/test.db",
        }
        (config_dir / "config.yaml").write_text(yaml.dump(yaml_data))

        # Temporarily point TENANTS_DIR to tmp_path
        import app.tenant as tenant_module
        original = tenant_module.TENANTS_DIR
        tenant_module.TENANTS_DIR = tmp_path / "tenants"
        try:
            config = TenantConfig.load(tid)
            assert config.name == "Test Hospital"
            assert config.kg_db_path == "/data/test.db"
        finally:
            tenant_module.TENANTS_DIR = original

    def test_load_missing_config_returns_defaults(self, tmp_path):
        import app.tenant as tenant_module
        original = tenant_module.TENANTS_DIR
        tenant_module.TENANTS_DIR = tmp_path / "tenants"
        try:
            config = TenantConfig.load("nonexistent_tenant")
            assert config.tenant_id == "nonexistent_tenant"
            assert config.name == "HealthyPartner"  # default
        finally:
            tenant_module.TENANTS_DIR = original

    def test_config_file_cannot_override_tenant_id(self, tmp_path):
        tid = "real_tenant"
        config_dir = tmp_path / "tenants" / tid
        config_dir.mkdir(parents=True)
        yaml_data = {"tenant_id": "injected_id", "name": "Legit"}
        (config_dir / "config.yaml").write_text(yaml.dump(yaml_data))

        import app.tenant as tenant_module
        original = tenant_module.TENANTS_DIR
        tenant_module.TENANTS_DIR = tmp_path / "tenants"
        try:
            config = TenantConfig.load(tid)
            assert config.tenant_id == "real_tenant"  # not overridden
        finally:
            tenant_module.TENANTS_DIR = original


# ── TenantManager ──────────────────────────────────────────────────────────────

class TestTenantManager:
    def test_get_orchestrator_creates_on_first_call(self, mock_engine, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        manager = TenantManager(engine=mock_engine)
        orch = manager.get_orchestrator(DEFAULT_TENANT_ID)
        assert orch is not None

    def test_get_orchestrator_returns_same_instance(self, mock_engine, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        manager = TenantManager(engine=mock_engine)
        orch1 = manager.get_orchestrator(DEFAULT_TENANT_ID)
        orch2 = manager.get_orchestrator(DEFAULT_TENANT_ID)
        assert orch1 is orch2

    def test_different_tenants_get_different_orchestrators(self, mock_engine, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        manager = TenantManager(engine=mock_engine)
        orch_a = manager.get_orchestrator("tenant_a")
        orch_b = manager.get_orchestrator("tenant_b")
        assert orch_a is not orch_b

    def test_reload_evicts_cache(self, mock_engine, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        manager = TenantManager(engine=mock_engine)
        orch1 = manager.get_orchestrator(DEFAULT_TENANT_ID)
        manager.reload_tenant(DEFAULT_TENANT_ID)
        orch2 = manager.get_orchestrator(DEFAULT_TENANT_ID)
        assert orch1 is not orch2
