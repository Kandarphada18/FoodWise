from fastapi import FastAPI
app=FastAPI(title='FoodWise API')
@app.get('/')
def root(): return {'name':'FoodWise API','status':'running','docs':'/docs'}
@app.get('/health')
def health(): return {'status':'ok'}
