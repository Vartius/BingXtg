import json
from channels.generic.websocket import AsyncWebsocketConsumer


class DashboardConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """Handles new WebSocket connections."""
        self.room_group_name = "dashboard"

        # Join the 'dashboard' group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):
        """Handles WebSocket disconnections."""
        # Leave the 'dashboard' group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    async def dashboard_update(self, event):
        """
        Receives a message from the channel layer group
        and forwards it to the client's WebSocket.
        """
        message = event["message"]
        await self.send(text_data=json.dumps({"message": message}))
