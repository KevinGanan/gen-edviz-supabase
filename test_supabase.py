import streamlit as st
from services.supabase_client import supabase

st.title("Prueba de conexión Supabase")

st.write("Probando conexión...")

response = supabase.table("imagenes").select("*").execute()

st.write(response)
