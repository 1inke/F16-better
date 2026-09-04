"""
ASTRA Experiment Email Notifier
负责在每轮实验完成并获得 ChatGPT 诊断后，将结构化简报发送至指定邮箱 (chenhuarong6@gmail.com)。
"""
from __future__ import annotations

import json
import logging
import smtplib
import sys
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("EmailNotifier")

CONFIG_PATH = Path(__file__).resolve().parent / "email_config.json"
DEFAULT_RECIPIENT = "chenhuarong6@gmail.com"


def load_email_config() -> Dict[str, Any]:
    """读取本地邮件配置"""
    if CONFIG_PATH.is_file():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"读取 email_config.json 失败: {e}")
    return {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "use_tls": True,
        "sender_email": "",
        "sender_password": "",  # Gmail 应用专用密码 (App Password) 或 授权码
        "recipient_email": DEFAULT_RECIPIENT
    }


def save_default_email_config():
    """初始化模板配置文件"""
    if not CONFIG_PATH.is_file():
        default_cfg = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "use_tls": True,
            "sender_email": "your_email@gmail.com",
            "sender_password": "your_app_password_here",
            "recipient_email": DEFAULT_RECIPIENT,
            "note": "若使用 Gmail，密码请在 Google 账号中生成 16 位'应用专用密码(App Password)'；也可以使用 QQ、163 或 Outlook 邮箱发信。"
        }
        CONFIG_PATH.write_text(json.dumps(default_cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_html_briefing(
    round_idx: int,
    status_str: str,
    commit_sha: str,
    artifacts: Dict[str, Any],
    decision: Dict[str, Any],
    response_text: str
) -> str:
    """生成排版精美的学术实验 HTML 简报"""
    result_md = artifacts.get("result_analysis", "（无独立的 RESULT_ANALYSIS.md）")
    # 提取决策关键字段
    dec_type = decision.get("decision", "未知")
    next_goal = decision.get("next_round_goal", "无")
    next_cmd = decision.get("recommended_command", "无")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #24292e; max-width: 800px; margin: 0 auto; padding: 20px; }}
  .header {{ background: #0969da; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
  .header h1 {{ margin: 0; font-size: 22px; }}
  .badge {{ display: inline-block; padding: 4px 10px; border-radius: 12px; font-weight: bold; background: white; color: #0969da; margin-top: 10px; }}
  .card {{ border: 1px solid #e1e4e8; border-radius: 8px; padding: 18px; margin-bottom: 20px; background: #ffffff; }}
  .card h2 {{ font-size: 17px; margin-top: 0; border-bottom: 2px solid #f1f3f5; padding-bottom: 8px; color: #1f2328; }}
  .metric-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
  .metric-table th, .metric-table td {{ border: 1px solid #d0d7de; padding: 8px 12px; text-align: left; font-size: 13px; }}
  .metric-table th {{ background-color: #f6f8fa; }}
  .code-box {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 12px; font-family: Consolas, monospace; font-size: 13px; white-space: pre-wrap; overflow-x: auto; }}
  .decision-box {{ background: #e6ffed; border: 1px solid #acf2bd; border-radius: 6px; padding: 14px; }}
  .footer {{ font-size: 12px; color: #57606a; text-align: center; margin-top: 30px; }}
</style>
</head>
<body>
  <div class="header">
    <h1>ASTRA 轨迹预测实验简报 (第 {round_idx} 轮)</h1>
    <div class="badge">执行状态：{status_str} | 下轮决策：{dec_type}</div>
  </div>

  <div class="card">
    <h2>1. 实验基本信息与云端同步</h2>
    <p><strong>代码仓库：</strong> <code>1inke/pedestrian-trajectory-research</code></p>
    <p><strong>最新 Commit：</strong> <code>{commit_sha}</code></p>
    <p><strong>下轮实验目标：</strong> {next_goal}</p>
    {f'<p><strong>下一轮命令：</strong></p><div class="code-box">{next_cmd}</div>' if next_cmd != "无" else ''}
  </div>

  <div class="card">
    <h2>2. 本地分析文件摘要 (RESULT_ANALYSIS.md)</h2>
    <div class="code-box">{result_md[:2000]}</div>
  </div>

  <div class="card">
    <h2>3. 网页端 ChatGPT (5.6 high) 深度诊断与建议</h2>
    <div class="decision-box">
      <strong>【核心裁决】:</strong> {dec_type} - {next_goal}
    </div>
    <div class="code-box" style="margin-top: 12px; max-height: 450px; overflow-y: auto;">
{response_text}
    </div>
  </div>

  <div class="footer">
    此邮件由 ASTRA 本地全自动实验调度器发出，已为您自动保留所有日志与快照。
  </div>
</body>
</html>"""
    return html


def send_experiment_email(
    round_idx: int,
    status_str: str,
    commit_sha: str,
    artifacts: Dict[str, Any],
    decision: Dict[str, Any],
    response_text: str,
    save_fallback_dir: Optional[Path] = None
) -> bool:
    """发送实验简报邮件，若未配置或发送失败则优雅 fallback 并存盘"""
    cfg = load_email_config()
    sender_email = cfg.get("sender_email", "").strip()
    sender_password = cfg.get("sender_password", "").strip()
    recipient_email = cfg.get("recipient_email", DEFAULT_RECIPIENT).strip()

    html_content = generate_html_briefing(
        round_idx, status_str, commit_sha, artifacts, decision, response_text
    )

    # 保存一份 HTML 简报在本地日志目录
    if save_fallback_dir:
        (save_fallback_dir / "briefing.html").write_text(html_content, encoding="utf-8")

    # 检查是否有有效发信账号配置
    if not sender_email or not sender_password or sender_email == "your_email@gmail.com":
        logger.info(
            f"[邮件通知] 本地 email_config.json 尚未配置发信账号/密码，简报已保存至: "
            f"{save_fallback_dir / 'briefing.html' if save_fallback_dir else '本地'}。\n"
            f"如需自动发到 {recipient_email}，请在 email_config.json 中填入发信邮箱和授权密码。"
        )
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = Header(f"[ASTRA实验简报] 第 {round_idx} 轮: {decision.get('decision', '报告')} - {status_str}", "utf-8")
        msg["From"] = sender_email
        msg["To"] = recipient_email

        # 挂载正文
        msg.attach(MIMEText(html_content, "html", "utf-8"))

        server_host = cfg.get("smtp_server", "smtp.gmail.com")
        server_port = int(cfg.get("smtp_port", 587))
        use_tls = cfg.get("use_tls", True)

        logger.info(f"[邮件通知] 正在连接 SMTP 服务器 {server_host}:{server_port} 发送简报至 {recipient_email}...")
        if use_tls:
            server = smtplib.SMTP(server_host, server_port, timeout=20)
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(server_host, server_port, timeout=20)

        server.login(sender_email, sender_password)
        server.sendmail(sender_email, [recipient_email], msg.as_string())
        server.quit()

        logger.info(f"✅ [邮件通知] 实验简报已成功发送至: {recipient_email}")
        return True
    except Exception as e:
        logger.warning(f"❌ [邮件通知] 发送邮件失败: {e}（简报已保存在本地）")
        return False


if __name__ == "__main__":
    save_default_email_config()
    print(f"已初始化 email_config.json，默认收件人: {DEFAULT_RECIPIENT}")
