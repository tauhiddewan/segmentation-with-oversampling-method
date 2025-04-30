import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import dotenv_values


def send_notification(subject, 
                      body,
                      sender_email, 
                      receiver_email, 
                      smtp_server, 
                      smtp_port, 
                      password, 
                      results_fname=None, 
                      results_fpath=None):

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))  

    # Check if file attachment is provided
    if results_fname and results_fpath:
        try:
            with open(results_fpath, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition", f"attachment; filename= {results_fname}"
                )
                msg.attach(part)
        except FileNotFoundError:
            return f"Error: File '{results_fname}' not found at path '{results_fpath}'."

    # Secure the connection and send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            return "Email sent successfully"
        
    except smtplib.SMTPException as e:
        return f"SMTP error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"