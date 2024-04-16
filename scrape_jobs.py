from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Setup WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Open a webpage
driver.get("https://www.deeprec.ai")

# Use XPath to find elements containing the word "job" in their text.
# This XPath looks for any element (*) that contains the text "job".
jobs = driver.find_elements(By.XPATH, "//*[contains(translate(text(), 'JOB', 'job'), 'job')]")

for job in jobs:
    print(job.text)

# Use XPath to find elements with attributes containing the word "job".
# This is a more complex and less common scenario, as you'd need to check multiple attributes.
# Here's an example for 'class' and 'id' attributes, but it can be adapted for others.
jobs_in_attributes = driver.find_elements(By.XPATH, "//*[@*[contains(translate(., 'JOB', 'job'), 'job')]]")

for job in jobs_in_attributes:
    print(job.get_attribute("outerHTML"))

# Close the browser
driver.quit()
