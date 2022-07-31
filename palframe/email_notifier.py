#!/usr/bin/env python
#coding: utf8

import smtplib
from palframe.nlp import Logger


class EmailNotifier:
  def __init__(self, recipient_emails: list = []):
    assert type(recipient_emails) is list
    self._recipients = recipient_emails

  def send_email(self, subject: str, text: str) -> bool:
    gmail_user = "xinzhikeji2015@gmail.com"
    gmail_pwd = "tianxia02092015"
    FROM = "xinzhikeji2015@gmail.com"
    TO = self._recipients

    # Prepare actual message
    message = f"\From: {FROM}\n" \
              f"To: {', '.join(TO)}\n" \
              f"Subject: {subject}\n\n" \
              f"{text}"
    try:
      server = smtplib.SMTP("smtp.gmail.com", 587)
      server.ehlo()
      server.starttls()
      server.login(gmail_user, gmail_pwd)
      server.sendmail(FROM, TO, message.encode("utf8"))
      server.close()
      return True

    except Exception as err:
      Logger.error(err)
      return False


if __name__ == "__main__":
  email_notifier = EmailNotifier(["SummerRainET2008@gmail.com"])

  status = email_notifier.send_email("你好", "测试自动email")
  if status:
    print("OK")
  else:
    print("Failed")
