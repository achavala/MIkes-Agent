# 🚀 Community Platform MVP - Setup Guide

## ✅ What's Built

### 1. Signal Integration ✅
- Mike Agent now generates signals automatically
- Entry and exit signals for all trades
- Integrated with `platform/integrate_signals.py`

### 2. Database Models ✅
- PostgreSQL schema ready
- Users, signals, subscriptions, performance tables
- Run `python platform/init_db.py` to initialize

### 3. FastAPI Backend ✅
- REST API endpoints
- WebSocket for real-time updates
- Stripe integration module

### 4. Next.js Dashboard ✅
- Professional dark theme
- Real-time signal feed
- Performance metrics
- WebSocket connection

### 5. Mobile App Structure ✅
- React Native/Expo setup
- Signal feed screen
- Performance screen
- Navigation ready

---

## 📋 Setup Steps

### Step 1: Install Platform Dependencies

```bash
cd platform
pip install -r requirements.txt
```

### Step 2: Set Up Database

**Option A: Railway PostgreSQL (Recommended)**
1. Go to Railway dashboard
2. Create new PostgreSQL service
3. Copy connection string
4. Set environment variable:
   ```bash
   export DATABASE_URL="postgresql://user:pass@host:port/dbname"
   ```

**Option B: Local PostgreSQL**
```bash
createdb community_platform
export DATABASE_URL="postgresql://localhost/community_platform"
```

### Step 3: Initialize Database

```bash
python platform/init_db.py
```

### Step 4: Start API Server

```bash
cd platform
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### Step 5: Set Up Stripe

1. Create Stripe account: https://stripe.com
2. Get API keys from dashboard
3. Create products for each tier:
   - Basic: $100/month
   - Pro: $200/month
   - Elite: $500/month
4. Set environment variables:
   ```bash
   export STRIPE_SECRET_KEY="sk_..."
   export STRIPE_PUBLISHABLE_KEY="pk_..."
   export STRIPE_WEBHOOK_SECRET="whsec_..."
   ```

### Step 6: Start Dashboard

```bash
cd dashboard
npm install
npm run dev
```

Visit: http://localhost:3000

### Step 7: Mobile App (Optional)

```bash
cd mobile-app
npm install
npx expo start
```

---

## 🔧 Environment Variables

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/dbname

# Stripe
STRIPE_SECRET_KEY=sk_...
STRIPE_PUBLISHABLE_KEY=pk_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_BASIC_PRICE_ID=price_...
STRIPE_PRO_PRICE_ID=price_...
STRIPE_ELITE_PRICE_ID=price_...

# API
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 🚀 Deployment

### Railway Deployment

1. **API Server:**
   - Create new service
   - Set build command: `pip install -r platform/requirements.txt`
   - Set start command: `uvicorn platform.api_server:app --host 0.0.0.0 --port $PORT`
   - Add environment variables

2. **Dashboard:**
   - Create new service
   - Set build command: `cd dashboard && npm install && npm run build`
   - Set start command: `cd dashboard && npm start`
   - Add environment variables

3. **Database:**
   - Create PostgreSQL service
   - Run migrations: `python platform/init_db.py`

---

## 📱 Next Steps

1. ✅ Test signal generation (run Mike Agent)
2. ✅ Test API endpoints
3. ✅ Test dashboard
4. ⏳ Add authentication
5. ⏳ Complete Stripe integration
6. ⏳ Add paper trading
7. ⏳ Deploy to production

---

## 🎯 Revenue Setup

### Stripe Products

Create these in Stripe Dashboard:
- **Basic:** $100/month (recurring)
- **Pro:** $200/month (recurring)
- **Elite:** $500/month (recurring)

### Pricing Page

Add subscription page to dashboard:
- Display tiers
- Stripe Checkout integration
- Success/cancel pages

---

## ✅ MVP Checklist

- [x] Signal service integrated
- [x] Database models created
- [x] API server structure
- [x] Dashboard UI
- [x] Mobile app structure
- [ ] Authentication system
- [ ] Stripe checkout
- [ ] Webhook handling
- [ ] Paper trading
- [ ] Production deployment

---

**Ready to launch! 🚀**
