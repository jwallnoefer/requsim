import unittest
from requsim.world import World
from requsim.events import EventQueue
from requsim.quantum_objects import WorldObject, Station, Source


class DummyObject(WorldObject):
    def __init__(self, world):
        super(DummyObject, self).__init__(world=world)


class CustomTypeObject(WorldObject):
    def __init__(self, world, custom_type):
        self._type = custom_type
        super(CustomTypeObject, self).__init__(world=world)

    @property
    def type(self):
        return self._type


class TestWorldSetup(unittest.TestCase):
    def setUp(self):
        self.world = World()

    def test_attributes(self):
        """Test for the existance of central attributes."""
        self.assertIsInstance(self.world.world_objects, dict)
        self.assertIsInstance(self.world.event_queue, EventQueue)

    def test_object_registration(self):
        """Test whether world objects are automatically registered."""
        test_station = Station(world=self.world, position=0)
        self.assertIn(test_station, self.world.world_objects[test_station.type])
        # repeat test with custom __contains__ method
        self.assertIn(test_station, self.world)
        test_source = Source(
            world=self.world, position=0, target_stations=[test_station, test_station]
        )
        self.assertIn(test_source, self.world.world_objects[test_source.type])
        # repeat test with custom __contains__ method
        self.assertIn(test_source, self.world)
        for i in range(20):
            test_quantum_object = DummyObject(world=self.world)
            self.assertIn(
                test_quantum_object, self.world.world_objects[test_quantum_object.type]
            )
            # repeat test with custom __contains__ method
            self.assertIn(test_quantum_object, self.world)

    def test_object_deregistration(self):
        test_station = Station(world=self.world, position=0)
        test_source = Source(
            world=self.world, position=0, target_stations=[test_station, test_station]
        )
        test_quantum_object = DummyObject(world=self.world)
        custom_type_object_1 = CustomTypeObject(
            world=self.world, custom_type="VeryCustomMuchWow"
        )
        custom_type_object_2 = CustomTypeObject(
            world=self.world, custom_type="VeryCustomMuchWow"
        )

        # now deregister
        self.assertIn(custom_type_object_1, self.world)
        custom_type_object_1.destroy()
        self.assertNotIn(
            custom_type_object_1, self.world.world_objects[custom_type_object_1.type]
        )
        self.assertNotIn(custom_type_object_1, self.world)
        self.assertIn(custom_type_object_2, self.world)
        self.world.deregister_world_object(custom_type_object_2)
        self.assertNotIn(
            custom_type_object_2, self.world.world_objects[custom_type_object_2.type]
        )
        self.assertNotIn(custom_type_object_2, self.world)
        self.assertIn(test_quantum_object, self.world)
        test_quantum_object.destroy()
        self.assertNotIn(test_quantum_object, self.world)
        self.assertIn(test_source, self.world)
        test_source.destroy()
        self.assertNotIn(test_source, self.world)
        self.assertIn(test_station, self.world)
        test_station.destroy()
        self.assertNotIn(test_station, self.world)

    def test_label_assignment(self):
        test_station = Station(world=self.world, position=0)
        self.assertEqual(test_station.label, test_station.type + " 1")
        self.assertEqual(self.world._label_counters[test_station.type], 1)
        custom_station_label = "The Great Station"
        test_station_2 = Station(
            world=self.world, position=0, label=custom_station_label
        )
        self.assertEqual(test_station_2.label, custom_station_label)
        self.assertEqual(self.world._label_counters[test_station.type], 2)
        for i in range(1, 21):
            test_quantum_object = DummyObject(world=self.world)
            self.assertEqual(
                test_quantum_object.label, test_quantum_object.type + " " + str(i)
            )
            custom_type_object = CustomTypeObject(
                world=self.world, custom_type="MyCustomType"
            )
            self.assertEqual(
                custom_type_object.label, custom_type_object.type + " " + str(i)
            )


if __name__ == "__main__":
    unittest.main()
