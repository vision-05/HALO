from discovery.src.base_agent import BaseAgent
from apyhiveapi import Auth, Hive
import os
import asyncio

user = os.environ.get("HIVE_USER")
passw = os.environ.get("HIVE_PASS")

class HiveAgent(BaseAgent):
    def __init__(self):
        super().__init__("HiveThermostat", "Actuator")
        self.register_handlers({"set_temperature": self.set_temperature,
                                "set_boost_temperature_with_duration": self.set_boost_temperature_with_duration,
                                "set_boost_off": self.set_boost_off})

    async def auth(self):
        tokens = None
        auth = Auth(username=user, password=passw)
        tokens = await auth.login()
        #code = input("SMS Code: ")
        #tokens = await auth.sms_2fa(code, tokens)

        self.hive = Hive(username=user, password=passw)
        await self.hive.startSession({"tokens": tokens})

        self.devices = [device for device in self.hive.session.data.devices.values()]


    async def set_temperature(self, msg: dict):
        temp = msg["params"].get("temp")
        await self.set_boost_off(msg)
        await self.hive.heating.set_target_temperature(self.devices[0], temp)
        self.state.update({"temp": temp})

    async def set_boost_temperature_with_duration(self, msg: dict):
        temp = msg["params"].get("temp")
        duration = msg["params"].get("duration")

        await self.hive.heating.set_boost_on(self.devices[0], mins=duration, temp=temp)
        self.state.update({"mode": "boost", "temp": temp})

    async def set_boost_off(self, msg: dict):
        await self.hive.heating.set_boost_off(self.devices[0]) #add temperature get
        self.state.update({"mode": "normal"})

async def main():
    thermo = HiveAgent()
    await thermo.auth()
    await thermo.run()

if __name__ == "__main__":
    asyncio.run(main())
