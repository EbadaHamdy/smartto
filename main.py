from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Load the DataFrame
df = pd.read_csv(r"https://raw.githubusercontent.com/EbadaHamdy/smartto/main/0.csv")


# Convert Tag, Review, and Comment columns to string type
df['Tag'] = df['Tag'].astype(str)
df['Review'] = df['Review'].astype(str)
df['Comment'] = df['Comment'].astype(str)

class PlanRequest(BaseModel):
    country: str
    governorates: List[str]
    survey_responses: List[str]
    num_days: int
    budget: float
    num_plans: int

class PlanRecommendation(BaseModel):
    hotel: str
    hotel_price_per_day: float
    total_hotel_price: float
    recommendations: List[str]
    total_plan_price: float
    additional_amount_needed: str

class RecommendationsResponse(BaseModel):
    plan_number: int
    hotel: str
    hotel_price_per_day: float
    total_hotel_price: float
    plan_recommendations: List[str]
    total_plan_price: float
    additional_amount_needed: str

# Feature Engineering: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Tag'] + ' ' + df['Review'] + ' ' + df['Comment'])

# Store recommendations globally
recommendations_storage = []

def get_recommendations_with_budget(country, governorates, survey_responses, num_days, budget, num_plans=1):
    if num_days > 7:
        raise HTTPException(status_code=400, detail="The maximum number of days allowed is 7.")

    filtered_df = df[(df['Country'] == country) & (df['Governorate'].isin(governorates))]

    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="No data found for the specified country and governorates.")

    all_recommendations = []

    for plan_num in range(1, num_plans + 1):
        user_profile = f"{country} {' '.join(governorates)} {' '.join(survey_responses)}"
        user_profile_vectorized = tfidf_vectorizer.transform([user_profile])
        max_price_per_day = budget / num_days
        recommendations_df = pd.DataFrame(columns=['Title', 'Price', 'tags', 'Governorate', 'Day'])
        recommended_titles = set()
        hotels = filtered_df[filtered_df['tags'].str.lower().str.contains('hotel') & (filtered_df['Price'] <= max_price_per_day)]
        hotel_recommendation = pd.DataFrame()
        hotel_price = 0.0

        if not hotels.empty:
            hotel_recommendation = hotels.sample(1)[['Title', 'Price', 'tags', 'Governorate']]
            recommended_titles.add(hotel_recommendation['Title'].iloc[0])
            hotel_price = hotel_recommendation['Price'].iloc[0] * num_days

        for day in range(1, num_days + 1):
            daily_recommendations = []
            governorate_df = filtered_df[filtered_df['Governorate'].isin(governorates)]
            restaurants = governorate_df[governorate_df['tags'].str.lower().str.contains('restaurant') & (governorate_df['Price'] <= max_price_per_day)]
            if not restaurants.empty:
                restaurant_recommendation = restaurants.sample(1)
                for _, row in restaurant_recommendation.iterrows():
                    row['Day'] = day
                    daily_recommendations.append(row)
                    recommended_titles.add(row['Title'])
            random.shuffle(survey_responses)
            place_recommendations = []
            for response in survey_responses:
                response_indices = [i for i, tag in enumerate(governorate_df['tags']) if response.lower() in tag.lower()]
                valid_indices = [idx for idx in response_indices if governorate_df.iloc[idx]['Title'] not in recommended_titles]
                if valid_indices:
                    random_index = random.choice(valid_indices)
                    recommendation = governorate_df.iloc[random_index][['Title', 'Price', 'tags', 'Governorate']]
                    recommendation['Day'] = day
                    place_recommendations.append(recommendation)
                    recommended_titles.add(recommendation['Title'])
            num_additional_recommendations = min(2, len(place_recommendations))
            daily_recommendations.extend(place_recommendations[:num_additional_recommendations])
            for recommendation in daily_recommendations:
                recommendations_df = pd.concat([recommendations_df, pd.DataFrame([recommendation])])

        total_plan_price = hotel_price
        for idx, row in recommendations_df.iterrows():
            total_plan_price += row['Price']

        additional_amount_needed = 0.0
        if total_plan_price > budget:
            additional_amount_needed = total_plan_price - budget

        if additional_amount_needed == 0.0:
            additional_amount_message = "You do not need to add any additional money for this trip."
        else:
            additional_amount_message = f"You have entered {budget}, and you need an additional {additional_amount_needed}."

        plan_recommendations = []
        for day in range(1, num_days + 1):
            day_recommendations = recommendations_df[recommendations_df['Day'] == day]
            if len(day_recommendations) >= 3:
                daily_plan = [
                    f"Day {day}:",
                    f"Restaurant: {day_recommendations.iloc[0]['Title']} → Price: {day_recommendations.iloc[0]['Price']}",
                    f"{day_recommendations.iloc[1]['tags']}: {day_recommendations.iloc[1]['Title']} → Price: {day_recommendations.iloc[1]['Price']}",
                    f"{day_recommendations.iloc[2]['tags']}: {day_recommendations.iloc[2]['Title']} → Price: {day_recommendations.iloc[2]['Price']}"
                ]
                plan_recommendations.extend(daily_plan)

        all_recommendations.append({
            'plan_number': plan_num,
            'hotel': hotel_recommendation['Title'].iloc[0] if not hotel_recommendation.empty else "No suitable hotel found",
            'hotel_price_per_day': hotel_recommendation['Price'].iloc[0] if not hotel_recommendation.empty else 0.0,
            'total_hotel_price': hotel_price,
            'plan_recommendations': plan_recommendations,
            'total_plan_price': total_plan_price,
            'additional_amount_needed': additional_amount_message,
            'country': country,
            'governorates': governorates,
            'survey_responses': survey_responses,
            'num_days': num_days,
            'budget': budget,
            'num_plans': num_plans
        })

    return all_recommendations

@app.post("/recommendations/", response_model=List[RecommendationsResponse])
def recommend_plans(request: PlanRequest):
    global recommendations_storage
    recommendations_storage = get_recommendations_with_budget(
        request.country,
        request.governorates,
        request.survey_responses,
        request.num_days,
        request.budget,
        request.num_plans
    )
    return recommendations_storage

@app.get("/plans/{plan_num}", response_model=RecommendationsResponse)
def get_plan(plan_num: int):
    global recommendations_storage

    if plan_num < 1 or plan_num > len(recommendations_storage):
        raise HTTPException(status_code=404, detail="Plan number out of range.")

    return recommendations_storage[plan_num - 1]

@app.put("/add-funds/{plan_num}", response_model=RecommendationsResponse)
def add_funds(plan_num: int, additional_funds: float):
    global recommendations_storage

    if plan_num < 1 or plan_num > len(recommendations_storage):
        raise HTTPException(status_code=404, detail="Plan number out of range.")

    current_plan = recommendations_storage[plan_num - 1]

    # Calculate the remaining funds after adding additional funds
    current_plan['budget'] += additional_funds
    remaining_funds = current_plan['budget'] - current_plan['total_plan_price']

    # Determine if additional funds are now sufficient
    if remaining_funds >= 0:
        additional_amount_message = "Funds added successfully. You have chosen this trip."
    else:
        additional_amount_needed = current_plan['total_plan_price'] - current_plan['budget']
        additional_amount_message = f"You have entered {additional_funds}, and you need an additional {additional_amount_needed}."

    # Construct message indicating how much funds were added and remaining funds
    message = f"You have added {additional_funds}. Remaining funds: {remaining_funds}."

    # Return updated plan information with the message
    return {
        "plan_number": plan_num,
        "hotel": current_plan['hotel'],
        "hotel_price_per_day": current_plan['hotel_price_per_day'],
        "total_hotel_price": current_plan['total_hotel_price'],
        "plan_recommendations": current_plan['plan_recommendations'],
        "total_plan_price": current_plan['total_plan_price'],
        "additional_amount_needed": f"{additional_amount_message} {message}",
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
