from openhands.app_server.sandbox.sandbox_router import router


def test_delete_route_binds_sandbox_id_path_parameter():
    delete_routes = [
        route
        for route in router.routes
        if getattr(route, 'methods', set()) == {'DELETE'}
    ]

    assert any(route.path == '/sandboxes/{sandbox_id}' for route in delete_routes)
