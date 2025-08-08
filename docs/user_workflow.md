This documents the user workflow to be implemented on Streamlit 

## User input 

Company name - as used in the APIs 

## Display 
The display will be broken into two tabs:
###  Filings download and pdf view 
1. Firstly use the filings dowload API to get the list of urls 
2. Then allow the user to choose the particular filing for pdf viewing in the streamlit main. 

### Financials view
1. User can select the following: 
- level of consolidation - Consolidated or Standalone
2. Three tabs update under the financials view for 
- statement_type - one of cash_flow, balance_sheet, profit_loss

Convert the json to table view doing an outer join on all key values pairs in each json element by level of consolidation and statement_type. 
