#!/bin/bash

echo "🚀 SimApp.ai - Enable Public Registration"
echo "========================================="

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}⚠️  WARNING: This will enable public user registration!${NC}"
echo ""
echo "This script will:"
echo "1. Re-enable backend registration endpoints"
echo "2. Re-enable Auth0 automatic user creation"
echo "3. Restore registration page functionality"
echo "4. Update landing page buttons to 'Start Free Trial'"
echo ""
read -p "Are you sure you want to enable public registration? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ Cancelled. Registration remains disabled.${NC}"
    exit 1
fi

echo -e "${BLUE}[INFO]${NC} Enabling public registration..."

# 1. Re-enable backend registration in auth router
echo -e "${BLUE}[INFO]${NC} Re-enabling backend auth registration..."
sed -i 's/raise HTTPException(/# raise HTTPException(/' backend/auth/router.py
sed -i 's/status_code=status.HTTP_403_FORBIDDEN,/# status_code=status.HTTP_403_FORBIDDEN,/' backend/auth/router.py
sed -i 's/detail="New user registrations are currently disabled."/# detail="New user registrations are currently disabled."/' backend/auth/router.py
sed -i 's/)/# )/' backend/auth/router.py
sed -i 's/# Original registration logic (now disabled again):/# Re-enable original registration logic:/' backend/auth/router.py
sed -i 's/# if user_in.password != user_in.password_confirm:/if user_in.password != user_in.password_confirm:/' backend/auth/router.py
sed -i 's/# db_user = auth_service.create_user(db=db, user_in=user_in)/db_user = auth_service.create_user(db=db, user_in=user_in)/' backend/auth/router.py
sed -i 's/# return db_user/return db_user/' backend/auth/router.py

# 2. Re-enable modular auth registration
echo -e "${BLUE}[INFO]${NC} Re-enabling modular auth registration..."
sed -i 's/raise HTTPException(/# raise HTTPException(/' backend/modules/auth/router.py
sed -i 's/status_code=status.HTTP_403_FORBIDDEN,/# status_code=status.HTTP_403_FORBIDDEN,/' backend/modules/auth/router.py
sed -i 's/detail="New user registrations are temporarily disabled. SimApp is currently in private launch mode."/# detail="New user registrations are temporarily disabled. SimApp is currently in private launch mode."/' backend/modules/auth/router.py
sed -i 's/)/# )/' backend/modules/auth/router.py
sed -i 's/# Original registration logic (disabled for private launch):/# Re-enable original registration logic:/' backend/modules/auth/router.py
sed -i 's/# return await auth_service.create_user(user_in.dict())/return await auth_service.create_user(user_in.dict())/' backend/modules/auth/router.py

# 3. Re-enable Auth0 user creation
echo -e "${BLUE}[INFO]${NC} Re-enabling Auth0 automatic user creation..."
# This is more complex, so we'll provide instructions
echo -e "${YELLOW}[MANUAL]${NC} You need to manually restore Auth0 user creation in:"
echo "         backend/auth/auth0_dependencies.py"
echo "         Uncomment the user creation logic in get_or_create_user_from_auth0()"

# 4. Restore registration page
echo -e "${BLUE}[INFO]${NC} Restoring registration page..."
sed -i 's/const PrivateLaunchPage = lazy(() => import(.\/pages\/PrivateLaunchPage.));/const RegisterPage = lazy(() => import(.\/pages\/RegisterPage.));/' frontend/src/App.jsx
sed -i 's/<PrivateLaunchPage \/>/<RegisterPage \/>/' frontend/src/App.jsx

# 5. Update landing page buttons
echo -e "${BLUE}[INFO]${NC} Updating landing page buttons..."
sed -i 's/Request Access/Start Free Trial/g' frontend/src/pages/LandingPage.jsx
sed -i 's/Learn More/Get Started/' frontend/src/pages/LandingPage.jsx

echo ""
echo -e "${GREEN}✅ Public registration has been enabled!${NC}"
echo ""
echo "Next steps:"
echo "1. Manually restore Auth0 user creation (see instructions above)"
echo "2. Restart containers: docker-compose -f docker-compose.domain.yml restart"
echo "3. Test registration flow"
echo ""
echo -e "${YELLOW}⚠️  Remember to update your marketing and launch communications!${NC}" 