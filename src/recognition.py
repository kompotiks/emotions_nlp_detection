import asyncio
import websockets
import wave


def recordiring_text(filename: str) -> list:
    text = []

    async def hello(uri):
        async with websockets.connect(uri) as websocket:
            wf = wave.open(filename, "rb")
            await websocket.send('''{"config" : 
                        { "word_list" : "zero one two three four five six seven eight nine oh",
                        "sample_rate" : 16000.0}}''')
            while True:
                data = wf.readframes(8000)

                if len(data) == 0:
                    break

                await websocket.send(data)
                words = await websocket.recv()
                print(words)
                text.append(words)

            await websocket.send('{"eof" : 1}')
            words = await websocket.recv()
            print(words)
            text.append(words)

    asyncio.get_event_loop().run_until_complete(
        hello('ws://localhost:2700'))
    return text
