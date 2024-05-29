"""Assert that the changes to how protocols are supposed to be initialized have the correct behavior."""

import pytest

from requsim.quantum_objects import Station, SchedulingSource
from requsim.tools.protocol import TwoLinkProtocol
from requsim.world import World
import requsim.libs.matrix as mat


class EmptyTwoLinkProtocol(TwoLinkProtocol):
    def check(self, message=None):
        pass


@pytest.fixture
def twolink_compatible_world():
    world = World()
    station_A = Station(world=world, position=0)
    station_central = Station(world=world, position=10)
    station_B = Station(world=world, position=20)
    source_A = SchedulingSource(
        world=world,
        position=station_central.position,
        target_stations=[station_A, station_central],
        time_distribution=lambda source: 2,
        state_generation=lambda source: mat.H(mat.phiplus) @ mat.phiplus,
    )
    source_B = SchedulingSource(
        world=world,
        position=station_central.position,
        target_stations=[station_central, station_B],
        time_distribution=lambda source: 2,
        state_generation=lambda source: mat.H(mat.phiplus) @ mat.phiplus,
    )
    return world


@pytest.fixture
def twolink_compatible_world2():
    world = World()
    station_A = Station(world=world, position=0)
    station_central = Station(world=world, position=10)
    station_B = Station(world=world, position=20)
    source_A = SchedulingSource(
        world=world,
        position=station_central.position,
        target_stations=[station_A, station_central],
        time_distribution=lambda source: 2,
        state_generation=lambda source: mat.H(mat.phiplus) @ mat.phiplus,
    )
    source_B = SchedulingSource(
        world=world,
        position=station_central.position,
        target_stations=[station_central, station_B],
        time_distribution=lambda source: 2,
        state_generation=lambda source: mat.H(mat.phiplus) @ mat.phiplus,
    )
    return world


def test_future_warning():
    with pytest.warns(FutureWarning):
        EmptyTwoLinkProtocol(world=World())
    with pytest.warns(FutureWarning):
        EmptyTwoLinkProtocol(communication_speed=123456789)
    with pytest.warns(FutureWarning):
        EmptyTwoLinkProtocol(World(), 2e8)


def test_new_setup(twolink_compatible_world):
    comm_speed = 2e8
    p = EmptyTwoLinkProtocol()
    p.setup(world=twolink_compatible_world, communication_speed=comm_speed)
    assert p.world == twolink_compatible_world
    assert p.communication_speed == comm_speed

    p = EmptyTwoLinkProtocol()
    with pytest.raises(ValueError):
        p.setup(world=twolink_compatible_world)

    p = EmptyTwoLinkProtocol()
    with pytest.raises(ValueError):
        p.setup(communication_speed=comm_speed)

    with pytest.warns(FutureWarning):
        p = EmptyTwoLinkProtocol(world=twolink_compatible_world)
    p.setup(communication_speed=comm_speed)
    assert p.world == twolink_compatible_world
    assert p.communication_speed == comm_speed

    with pytest.warns(FutureWarning):
        p = EmptyTwoLinkProtocol(communication_speed=comm_speed)
    p.setup(world=twolink_compatible_world)
    assert p.world == twolink_compatible_world
    assert p.communication_speed == comm_speed


def test_setup_takes_precedence(twolink_compatible_world, twolink_compatible_world2):
    # if conflicting information is passed, setup should take precedence
    comm_speed1 = 1234
    comm_speed2 = 2e8
    with pytest.warns(FutureWarning):
        p = EmptyTwoLinkProtocol(
            world=twolink_compatible_world, communication_speed=comm_speed1
        )
    p.setup(world=twolink_compatible_world2, communication_speed=comm_speed2)
    assert p.world != twolink_compatible_world
    assert p.world == twolink_compatible_world2
    assert p.communication_speed != comm_speed1
    assert p.communication_speed == comm_speed2
