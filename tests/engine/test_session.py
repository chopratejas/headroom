from headroom.engine.session import derive_session_key

SALT = b"headroom-session-v1"


def test_deterministic():
    assert derive_session_key(
        credential="sk-ant-123", conversation_scope="conv_9", salt=SALT
    ) == derive_session_key(credential="sk-ant-123", conversation_scope="conv_9", salt=SALT)


def test_tenant_isolation():
    assert derive_session_key(
        credential="sk-ant-A", conversation_scope=None, salt=SALT
    ) != derive_session_key(credential="sk-ant-B", conversation_scope=None, salt=SALT)


def test_conversation_scope_separates_sessions_for_same_tenant():
    assert derive_session_key(
        credential="sk", conversation_scope="conv_1", salt=SALT
    ) != derive_session_key(credential="sk", conversation_scope="conv_2", salt=SALT)


def test_none_scope_falls_back_to_tenant_only_and_is_stable():
    assert derive_session_key(
        credential="sk", conversation_scope=None, salt=SALT
    ) == derive_session_key(credential="sk", conversation_scope=None, salt=SALT)


def test_raw_secret_is_never_present_in_key():
    key = derive_session_key(credential="sk-ant-SUPERSECRET", conversation_scope=None, salt=SALT)
    assert "SUPERSECRET" not in key
    assert "sk-ant" not in key


def test_empty_conversation_scope_behaves_like_none():
    assert derive_session_key(
        credential="sk", conversation_scope="", salt=SALT
    ) == derive_session_key(credential="sk", conversation_scope=None, salt=SALT)


def test_derive_session_key_exported_from_engine_package():
    from headroom.engine import derive_session_key as exported

    assert exported is derive_session_key
