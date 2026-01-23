from setuptools import setup

setup(
    name='gym_multi_car_racing',
    version='1.0.1',
    url='https://github.com/igilitschenski/multi_car_racing',
    description='Gym Multi Car Racing Environment',
    packages=['gym_multi_car_racing'],
    install_requires=[
    "shapely",

    "Box2D==2.3.10",
	"box2d_py==2.3.5",
	"gymnasium==0.29.1",

	"pygame==2.5.2",
	"pyglet==1.5.0",
	"setuptools==68.2.2"
	

    ]
)
