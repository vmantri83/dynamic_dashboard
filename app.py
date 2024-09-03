import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
import mysql.connector  

from flask import Flask, request, jsonify, g
from mysql.connector import pooling

from flask_cors import CORS

app = Flask(__name__)

cors = CORS(app, origins= "*")

# Global variable to store the database connection pool
db_pool = None







# Define the path to the saved model
model_path = "E:/TFRetrained/conversewithh4b/lora_model"

# Load the tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Load the LoRA model
model = PeftModel.from_pretrained(model, model_path)





# Function to create the database connection pool
def create_db_pool(host, user, password, database, pool_size=5):
    global db_pool
    db_config = {
        "host": host,
        "user": user,
        "password": password,
        "database": database
        #"port":3306
    }
    db_pool = pooling.MySQLConnectionPool(pool_name="mypool",
                                                          pool_size=pool_size,
                                                          **db_config)
    


def get_db_connection():
    if 'db_connection' not in g:
        g.db_connection = db_pool.get_connection()
    return g.db_connection


# Route to handle the POST request to connect to the database
@app.route('/connect_old', methods=['POST'])
def connect_to_database_old():
    try:
        data = request.json
        host = data['host']
        user = data['user']
        password = data['password']
        database = data['database']

        create_db_pool(host, user, password, database)


        return jsonify({"message": "Connected to the database successfully!"}), 200

        
    except Exception as e:
        return jsonify({"error": f"Error connecting to the database: {e}"}), 500
    





def gen_response(user_question, schema):
    # Define your prompt
    # user_question = "Display the name of top product by price"
    prompt_template = f"""
    You are an SQL expert. Write MySQL queries according to user's request
    ### Context:
    '{schema}'
    

    ### Instruction :
    {user_question}
    ### Response:
    
    """

    # Tokenize the input
    inputs = tokenizer([prompt_template], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)

    # Decode and print the output
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(generated_text)
    print("\n\n\n\n")

    text = generated_text

    # Regular expression to extract the SQL query
    pattern = r'# Response:\n(.*?)\n</s>'

    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)

    sql_query = ""

    if match:
        sql_query = match.group(1).strip()
        print(sql_query)
    else:
        print("No SQL query found.")



    return sql_query










#Function to get table schemas from the database
def get_table_schemas_mysql():
    connection = get_db_connection()
    cursor = connection.cursor()
    table_names = get_table_names(cursor)
    table_schemas = {}
    try:
        counter = 0  # Initialize a counter
        for table_name in table_names:
            if counter >= 10:  # Check if counter exceeds 10
                break
            cursor.execute(f"SHOW CREATE TABLE {table_name}")
            schema = cursor.fetchone()
            table_schemas[table_name] = schema[1]
            counter += 1  # Increment the counter
        return table_schemas
    except Exception as e:
        print(f"Error getting table schemas: {e}")
        return {}






# Route to handle the POST request to generate a response based on a user query
@app.route('/generate_response', methods=['POST'])
def generate_response():
    try:
        # Get the user query from the request data
        user_input = request.json.get('query')
        # Ensure a query was provided
        if not user_input:
            return jsonify({"error": "No query provided"}), 400

        table_schemas = get_table_schemas_mysql()
        print(table_schemas)
        table_schemas_str = ""
        count = 0
        for table_name, schema in table_schemas.items():
            count+=1
            table_schemas_str += f"Table: {count}. {table_name}\nSchema: {schema}\n\n"
        table_schemas_str += f"There are total of {count} tables in database" 

        sql_query = gen_response(f"{user_input}", table_schemas_str)
        return jsonify({"message":f"{execute_query(sql_query)}", "sql":f"{sql_query}"})

    except Exception as e:
        return jsonify({"error": f"Error executing query: {e}"}), 500
    









def get_table_names(cursor):
    try:
        connection = get_db_connection()
        print(connection)
        cursor = connection.cursor()
        print(cursor)
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        cursor.close()
        return [table[0] for table in tables]
    except Exception as e:
        print(f"Error getting table names: {e}")
        return []
    


# Function to connect to the MySQL database
def connect_to_mysql_database(host, user, password, database):
    try:
        # Print debug information
        print(f"Attempting to connect to MySQL database at {host} with user {user}")
        
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        print("Connected to the database successfully!")
        return True, connection, connection.cursor()
    except mysql.connector.Error as err:
        print(f"Error connecting to the database: {err}")
        return False, None, None



# Function to execute a query on the database
def execute_query(query):
    try:
        connection = get_db_connection()

        cursor = connection.cursor()
        cursor.execute(query)
        # cursor = st.session_state['db_cursor']
        
        # if query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
        #     st.session_state['db_connection'].commit()
        
        # Check if the query is an INSERT statement
        if query.strip().upper().startswith("INSERT"):
            return None, None  # No column names or data to return for INSERT queries
        else:
            column_names = [desc[0] for desc in cursor.description]  # Get column names
            data = cursor.fetchall()  # Get data
            if column_names or data:
                return column_names, data
            else:
                return None, None
    
    except Exception as e:
        # st.session_state['history'].append((user_input, f"Can't execute query: {query}"))
        # st.session_state['past'].append(user_input)
        # st.session_state['generated'].append(f"Can't execute query:\n```sql\n{query}\n```\n")
        # st.error(f"Error executing query {e}")
        return None, None
    



# isCon, conn, cur = connect_to_mysql_database("localhost", "root", "user5000", "retaildb")


# if isCon:
#     table_schemas = get_table_schemas_mysql(cur)
#     table_schemas_str = ""
#     count = 0
#     for table_name, schema in table_schemas.items():
#         count+=1
#         table_schemas_str += f"Table: {count}. {table_name}\nSchema: {schema}\n\n"
#     table_schemas_str += f"There are total of {count} tables in database" 

#     sql_query = gen_response("Name the product name has the largest stock in my database", table_schemas_str)








if __name__ == "__main__":
    app.run()

    
