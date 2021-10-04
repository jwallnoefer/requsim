import unittest
from requsim.world import World
from requsim.events import EventQueue
from requsim.quantum_objects import WorldObject, Station, Source


class DummyObject(WorldObject):
    def __init__(self, world):
        super(DummyObject, self).__init__(world=world)


class TestWorldSetup(unittest.TestCase):
    def setUp(self):
        self.world = World()

    def test_attributes(self):
        """Test for the existance of central attributes."""
        self.assertIsInstance(self.world.world_objects, dict)
        self.assertIsInstance(self.world.event_queue, EventQueue)

    def test_contains_method(self):
        """Test whether the __contains__ magic method works as expected."""
        test_station = Station(world=self.world, position=0)
        self.assertTrue(test_station in self.world)
        test_source = Source(
            world=self.world, position=0, target_stations=[test_station, test_station]
        )
        self.assertTrue(test_source in self.world)
        test_quantum_object = DummyObject(world=self.world)
        self.assertTrue(test_quantum_object in self.world)


if __name__ == "__main__":
    unittest.main()
