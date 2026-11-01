import os, io, json
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext
from jose import jwt, JWTError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DB_PATH=os.getenv("SQLITE_PATH",os.path.join(BASE_DIR,"foodwise.db"))
os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)),exist_ok=True)
engine=create_engine("sqlite:///"+os.path.abspath(DB_PATH).replace("\\","/"),connect_args={"check_same_thread":False})
SessionLocal=sessionmaker(bind=engine,autoflush=False,autocommit=False)
Base=declarative_base()
pwd=CryptContext(schemes=["bcrypt"],deprecated="auto")
SECRET=os.getenv("JWT_SECRET","dev-secret-change-before-production")
security=HTTPBearer(auto_error=False)
MODEL=os.path.join(BASE_DIR,"ml_artifacts","model.joblib")
METRICS=os.path.join(BASE_DIR,"ml_artifacts","metrics.json")

class User(Base):
 __tablename__="users"
 id=Column(Integer,primary_key=True); name=Column(String(120),nullable=False); email=Column(String(255),unique=True,index=True,nullable=False); password_hash=Column(String(255),nullable=False); restaurant_name=Column(String(160),default="My Restaurant"); created_at=Column(DateTime,default=datetime.utcnow)
class Record(Base):
 __tablename__="records"
 id=Column(Integer,primary_key=True); user_id=Column(Integer,ForeignKey("users.id"),index=True); record_date=Column(Date,nullable=False); food_category=Column(String(100),default="General"); quantity_prepared=Column(Float,nullable=False); quantity_sold=Column(Float,nullable=False); actual_waste=Column(Float,nullable=False); customer_count=Column(Integer,default=0); average_daily_demand=Column(Float,default=0); previous_waste=Column(Float,default=0); season=Column(String(30),default="Unknown"); special_event=Column(Boolean,default=False)
class Prediction(Base):
 __tablename__="predictions"
 id=Column(Integer,primary_key=True); user_id=Column(Integer,ForeignKey("users.id"),index=True); record_date=Column(Date,nullable=False); food_category=Column(String(100)); quantity_prepared=Column(Float); quantity_sold=Column(Float); predicted_waste=Column(Float); waste_percentage=Column(Float); estimated_loss=Column(Float); created_at=Column(DateTime,default=datetime.utcnow)
Base.metadata.create_all(engine)
app=FastAPI(title="FoodWise API",version="1.0.0")
origins=[x.strip() for x in os.getenv("FRONTEND_ORIGINS","http://localhost:5173,https://foodwise-restaurant-waste.vercel.app").split(",") if x.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
def db_session():
 db=SessionLocal()
 try: yield db
 finally: db.close()
def token(user): return jwt.encode({"sub":str(user.id),"exp":datetime.utcnow()+timedelta(days=7)},SECRET,algorithm="HS256")
def whoami(creds:HTTPAuthorizationCredentials=Depends(security),db:Session=Depends(db_session)):
 if not creds: raise HTTPException(401,"Please sign in")
 try: uid=int(jwt.decode(creds.credentials,SECRET,algorithms=["HS256"])["sub"])
 except (JWTError,ValueError,KeyError): raise HTTPException(401,"Invalid session")
 u=db.query(User).filter(User.id==uid).first()
 if not u: raise HTTPException(401,"Account not found")
 return u
class Auth(BaseModel):
 email:str
 password:str=Field(min_length=8)
 name:str|None=None
 restaurant_name:str|None="My Restaurant"
class RecordIn(BaseModel):
 record_date:date
 food_category:str="General"
 quantity_prepared:float=Field(ge=0)
 quantity_sold:float=Field(ge=0)
 actual_waste:float=Field(ge=0)
 customer_count:int=Field(default=0,ge=0)
 average_daily_demand:float=Field(default=0,ge=0)
 previous_waste:float=Field(default=0,ge=0)
 season:str="Unknown"
 special_event:bool=False
class PredictIn(BaseModel):
 record_date:date
 food_category:str="General"
 quantity_prepared:float=Field(ge=0)
 quantity_sold:float=Field(ge=0)
 customer_count:int=Field(default=0,ge=0)
 average_daily_demand:float=Field(default=0,ge=0)
 previous_waste:float=Field(default=0,ge=0)
 season:str="Unknown"
 special_event:bool=False
 cost_per_kg:float=Field(default=120,ge=0)
FEATURES=["quantity_prepared","quantity_sold","customer_count","average_daily_demand","previous_waste","food_category","season","special_event","day_of_week"]
NUM=["quantity_prepared","quantity_sold","customer_count","average_daily_demand","previous_waste","day_of_week"]
CAT=["food_category","season","special_event"]
def features(r):
 return {"quantity_prepared":r.quantity_prepared,"quantity_sold":r.quantity_sold,"customer_count":r.customer_count or 0,"average_daily_demand":r.average_daily_demand or 0,"previous_waste":r.previous_waste or 0,"food_category":r.food_category or "General","season":r.season or "Unknown","special_event":bool(r.special_event),"day_of_week":r.record_date.weekday()}
def record_out(r):
 return {"id":r.id,"record_date":r.record_date.isoformat(),"food_category":r.food_category,"quantity_prepared":r.quantity_prepared,"quantity_sold":r.quantity_sold,"actual_waste":r.actual_waste,"customer_count":r.customer_count,"average_daily_demand":r.average_daily_demand,"previous_waste":r.previous_waste,"season":r.season,"special_event":r.special_event}
@app.get("/")
def root(): return {"name":"FoodWise API","status":"running","docs":"/docs"}
@app.get("/health")
def health(): return {"status":"ok"}
@app.post("/api/auth/register")
def register(x:Auth,db:Session=Depends(db_session)):
 email=x.email.strip().lower()
 if db.query(User).filter(User.email==email).first(): raise HTTPException(409,"Email already registered")
 u=User(name=(x.name or email.split("@")[0]),email=email,password_hash=pwd.hash(x.password),restaurant_name=x.restaurant_name or "My Restaurant")
 db.add(u);db.commit();db.refresh(u)
 return {"access_token":token(u),"token_type":"bearer","user":{"id":u.id,"name":u.name,"email":u.email,"restaurant_name":u.restaurant_name}}
@app.post("/api/auth/login")
def login(x:Auth,db:Session=Depends(db_session)):
 u=db.query(User).filter(User.email==x.email.strip().lower()).first()
 if not u or not pwd.verify(x.password,u.password_hash): raise HTTPException(401,"Email or password is incorrect")
 return {"access_token":token(u),"token_type":"bearer","user":{"id":u.id,"name":u.name,"email":u.email,"restaurant_name":u.restaurant_name}}
@app.get("/api/auth/me")
def me(u:User=Depends(whoami)): return {"id":u.id,"name":u.name,"email":u.email,"restaurant_name":u.restaurant_name}
@app.post("/api/records")
def add_record(x:RecordIn,db:Session=Depends(db_session),u:User=Depends(whoami)):
 if x.quantity_sold>x.quantity_prepared: raise HTTPException(400,"Sold quantity cannot exceed prepared quantity")
 r=Record(user_id=u.id,**x.model_dump());db.add(r);db.commit();db.refresh(r);return record_out(r)
@app.get("/api/records")
def get_records(db:Session=Depends(db_session),u:User=Depends(whoami)):
 return [record_out(r) for r in db.query(Record).filter(Record.user_id==u.id).order_by(Record.record_date.desc()).all()]
@app.delete("/api/records/{rid}")
def del_record(rid:int,db:Session=Depends(db_session),u:User=Depends(whoami)):
 r=db.query(Record).filter(Record.id==rid,Record.user_id==u.id).first()
 if not r: raise HTTPException(404,"Record not found")
 db.delete(r);db.commit();return {"message":"Record deleted"}
@app.post("/api/records/import")
async def import_csv(file:UploadFile=File(...),db:Session=Depends(db_session),u:User=Depends(whoami)):
 if not (file.filename or "").lower().endswith(".csv"): raise HTTPException(400,"Upload a CSV file")
 try:
  frame=pd.read_csv(io.BytesIO(await file.read()))
  needed={"quantity_prepared","quantity_sold","actual_waste"}
  if not needed.issubset(frame.columns): raise ValueError("CSV needs quantity_prepared, quantity_sold and actual_waste columns")
  n=0
  for _,v in frame.iterrows():
   prepared,sold,waste=map(float,[v.quantity_prepared,v.quantity_sold,v.actual_waste])
   if min(prepared,sold,waste)<0 or sold>prepared: continue
   dt=pd.to_datetime(v.get("record_date",date.today()),errors="coerce")
   db.add(Record(user_id=u.id,record_date=dt.date() if not pd.isna(dt) else date.today(),food_category=str(v.get("food_category","General")),quantity_prepared=prepared,quantity_sold=sold,actual_waste=waste,customer_count=int(v.get("customer_count",0) or 0),average_daily_demand=float(v.get("average_daily_demand",0) or 0),previous_waste=float(v.get("previous_waste",0) or 0),season=str(v.get("season","Unknown")),special_event=str(v.get("special_event",False)).lower() in ["1","true","yes"]));n+=1
  db.commit();return {"imported":n}
 except Exception as e:
  db.rollback();raise HTTPException(400,str(e))
@app.get("/api/records/export")
def export_csv(db:Session=Depends(db_session),u:User=Depends(whoami)):
 rs=db.query(Record).filter(Record.user_id==u.id).order_by(Record.record_date.asc()).all()
 return {"rows":[{k:v for k,v in record_out(r).items() if k!="id"} for r in rs]}
@app.post("/api/model/train")
def train(db:Session=Depends(db_session),u:User=Depends(whoami)):
 rs=db.query(Record).filter(Record.user_id==u.id).order_by(Record.record_date.asc()).all()
 if len(rs)<30: raise HTTPException(400,f"At least 30 labelled records required; found {len(rs)}")
 df=pd.DataFrame([{**features(r),"target":r.actual_waste} for r in rs]);X=df[FEATURES];y=df.target
 cut=max(1,min(len(df)-1,int(len(df)*.8)));Xtr,Xte=X.iloc[:cut],X.iloc[cut:];ytr,yte=y.iloc[:cut],y.iloc[cut:]
 prep=ColumnTransformer([("num",SimpleImputer(strategy="median"),NUM),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),CAT)])
 model=Pipeline([("prep",prep),("rf",RandomForestRegressor(n_estimators=200,random_state=42,min_samples_leaf=2))]);model.fit(Xtr,ytr);yp=np.maximum(0,model.predict(Xte))
 m={"mae":round(float(mean_absolute_error(yte,yp)),3),"rmse":round(float(np.sqrt(mean_squared_error(yte,yp))),3),"r2":round(float(r2_score(yte,yp)),3) if len(yte)>1 else None,"training_records":len(Xtr),"testing_records":len(Xte),"model":"Random Forest Regressor","trained":True}
 os.makedirs(os.path.dirname(MODEL),exist_ok=True);joblib.dump({"model":model,"user_id":u.id},MODEL)
 with open(METRICS,"w") as f: json.dump({"user_id":u.id,**m},f)
 return {"metrics":m}
@app.get("/api/model/performance")
def performance(u:User=Depends(whoami)):
 if not os.path.exists(METRICS): return {"trained":False,"message":"Add at least 30 labelled records and train the model."}
 with open(METRICS) as f:m=json.load(f)
 if m.get("user_id")!=u.id:return {"trained":False,"message":"No model trained for this account."}
 return m
@app.post("/api/predict")
def predict(x:PredictIn,db:Session=Depends(db_session),u:User=Depends(whoami)):
 if not os.path.exists(MODEL): raise HTTPException(400,"Train the model with at least 30 real records first")
 saved=joblib.load(MODEL)
 if saved.get("user_id")!=u.id:raise HTTPException(403,"Train your own model before predicting")
 row={**x.model_dump(),"day_of_week":x.record_date.weekday()};row.pop("record_date");row.pop("cost_per_kg")
 waste=float(np.clip(saved["model"].predict(pd.DataFrame([row])[FEATURES])[0],0,x.quantity_prepared))
 pct=round(waste/x.quantity_prepared*100,2) if x.quantity_prepared else 0
 p=Prediction(user_id=u.id,record_date=x.record_date,food_category=x.food_category,quantity_prepared=x.quantity_prepared,quantity_sold=x.quantity_sold,predicted_waste=round(waste,3),waste_percentage=pct,estimated_loss=round(waste*x.cost_per_kg,2));db.add(p);db.commit();db.refresh(p)
 return {"id":p.id,"predicted_waste":p.predicted_waste,"waste_percentage":pct,"estimated_loss":p.estimated_loss,"waste_level":"Low" if pct<5 else "Moderate" if pct<15 else "High","model":"Random Forest Regressor"}
@app.get("/api/predictions")
def predictions(db:Session=Depends(db_session),u:User=Depends(whoami)):
 return [{"id":p.id,"record_date":p.record_date.isoformat(),"food_category":p.food_category,"quantity_prepared":p.quantity_prepared,"predicted_waste":p.predicted_waste,"waste_percentage":p.waste_percentage,"estimated_loss":p.estimated_loss} for p in db.query(Prediction).filter(Prediction.user_id==u.id).order_by(Prediction.created_at.desc()).all()]
@app.get("/api/dashboard/summary")
def summary(db:Session=Depends(db_session),u:User=Depends(whoami)):
 rs=db.query(Record).filter(Record.user_id==u.id).all();ps=db.query(Prediction).filter(Prediction.user_id==u.id).all()
 prep=sum(r.quantity_prepared for r in rs);sold=sum(r.quantity_sold for r in rs);waste=sum(r.actual_waste for r in rs)
 return {"total_prepared":round(prep,2),"total_sold":round(sold,2),"total_waste":round(waste,2),"waste_percentage":round(waste/prep*100,2) if prep else 0,"records":len(rs),"predictions":len(ps)}
@app.get("/api/analytics")
def analytics(db:Session=Depends(db_session),u:User=Depends(whoami)):
 rs=db.query(Record).filter(Record.user_id==u.id).order_by(Record.record_date.asc()).all();daily={};cats={}
 for r in rs:daily[r.record_date.isoformat()]=daily.get(r.record_date.isoformat(),0)+r.actual_waste;cats[r.food_category]=cats.get(r.food_category,0)+r.actual_waste
 return {"daily":[{"date":k,"waste":round(v,2)} for k,v in daily.items()],"categories":[{"category":k,"waste":round(v,2)} for k,v in cats.items()]}
@app.get("/api/recommendations")
def recommendations(db:Session=Depends(db_session),u:User=Depends(whoami)):
 rs=db.query(Record).filter(Record.user_id==u.id).all();groups={}
 for r in rs:groups.setdefault(r.food_category,[]).append(r)
 items=[]
 for cat,arr in groups.items():
  prep=sum(r.quantity_prepared for r in arr);waste=sum(r.actual_waste for r in arr);rate=waste/prep*100 if prep else 0
  if rate>=10:items.append({"category":cat,"waste_rate":round(rate,1),"advice":f"Review batch sizes for {cat}; historical waste is {rate:.1f}% of prepared quantity."})
 return {"items":items,"message":"Recommendations use saved restaurant records." if items else "No category has exceeded the 10% waste threshold in saved records."}
