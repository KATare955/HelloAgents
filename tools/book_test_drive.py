"""试驾预约工具"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import random
import string
from hello_agents.tools import Tool, ToolParameter, ToolResponse


class BookTestDriveTool(Tool):
    """试驾预约工具
    
    预约附近经销商试驾时间。
    创建试驾预约记录，返回预约编号和确认信息。
    """
    
    def __init__(self):
        super().__init__(
            name="book_test_drive",
            description="预约附近经销商试驾时间，创建试驾预约记录，返回预约编号和确认信息"
        )
        # 加载Mock数据
        self.mock_data = self._load_mock_data()
    
    def _load_mock_data(self) -> Dict[str, Any]:
        """加载Mock数据"""
        mock_file = Path(__file__).parent / "mock_data" / "book_test_drive.json"
        with open(mock_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_parameters(self) -> List[ToolParameter]:
        """定义工具参数"""
        return [
            ToolParameter(
                name="car_series",
                type="string",
                description="试驾车系名称。",
                required=True
            ),
            ToolParameter(
                name="dealer",
                type="string",
                description="经销商名称。",
                required=True
            ),
            ToolParameter(
                name="appointment_time",
                type="string",
                description="预约时间，ISO 8601格式：YYYY-MM-DDTHH:MM:SS。需要在工作时间内（9:00-18:00）。",
                required=True
            ),
            ToolParameter(
                name="customer_name",
                type="string",
                description="客户姓名。",
                required=True
            ),
            ToolParameter(
                name="customer_phone",
                type="string",
                description="客户手机号，11位数字。",
                required=True
            ),
            ToolParameter(
                name="notes",
                type="string",
                description="备注信息，如特殊需求。",
                required=False,
                default=""
            )
        ]
    
    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """执行工具"""
        # 1. 获取参数
        car_series = parameters.get("car_series", "")
        dealer = parameters.get("dealer", "")
        appointment_time = parameters.get("appointment_time", "")
        customer_name = parameters.get("customer_name", "")
        customer_phone = parameters.get("customer_phone", "")
        notes = parameters.get("notes", "")
        
        # 2. 参数验证
        if not car_series:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="car_series参数不能为空"
            )
        
        if not dealer:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="dealer参数不能为空"
            )
        
        if not appointment_time:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="appointment_time参数不能为空"
            )
        
        if not customer_name:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="customer_name参数不能为空"
            )
        
        if not customer_phone:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="customer_phone参数不能为空"
            )
        
        # 验证手机号格式
        if not customer_phone.isdigit() or len(customer_phone) != 11:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="customer_phone必须是11位数字"
            )
        
        # 验证时间格式
        try:
            dt = datetime.fromisoformat(appointment_time.replace("Z", "+00:00"))
            # 检查是否在工作时间内（9:00-18:00）
            if dt.hour < 9 or dt.hour >= 18:
                return ToolResponse.error(
                    code="INVALID_PARAM",
                    message="预约时间需要在工作时间内（9:00-18:00）"
                )
        except ValueError:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="appointment_time格式错误，请使用ISO 8601格式：YYYY-MM-DDTHH:MM:SS"
            )
        
        # 3. 检查是否已有预约（Mock数据中查找）
        if car_series in self.mock_data:
            if dealer in self.mock_data[car_series]:
                if appointment_time in self.mock_data[car_series][dealer]:
                    # 返回已有的预约信息
                    existing_booking = self.mock_data[car_series][dealer][appointment_time]
                    return ToolResponse.success(
                        text=f"预约成功！预约编号：{existing_booking['booking_id']}",
                        data=existing_booking
                    )
        
        # 4. 创建新的预约记录
        booking_id = self._generate_booking_id()
        
        booking_data = {
            "booking_id": booking_id,
            "status": "confirmed",
            "car_series": car_series,
            "dealer": dealer,
            "appointment_time": appointment_time,
            "customer_name": customer_name,
            "customer_phone": customer_phone,
            "notes": notes,
            "confirmation": {
                "message": "预约成功，请准时到达",
                "reminder": "请携带驾驶证原件，提前15分钟到店",
                "contact": self._get_dealer_phone(dealer)
            },
            "created_at": datetime.now().isoformat() + "Z"
        }
        
        # 5. 返回结果
        return ToolResponse.success(
            text=f"预约成功！{customer_name}，您已预约{car_series}试驾，时间：{appointment_time}，地点：{dealer}",
            data=booking_data
        )
    
    def _generate_booking_id(self) -> str:
        """生成预约编号"""
        # 格式：TD + 日期 + 3位随机数
        date_str = datetime.now().strftime("%Y%m%d")
        random_str = ''.join(random.choices(string.digits, k=3))
        return f"TD{date_str}{random_str}"
    
    def _get_dealer_phone(self, dealer: str) -> str:
        """获取经销商电话（Mock）"""
        dealer_phones = {
            "上海郑隆奥翟": "021-12345678",
            "上海浦东奥迪": "021-87654321",
            "上海小米汽车体验中心": "021-11112222"
        }
        return dealer_phones.get(dealer, "400-000-0000")


# 测试代码
if __name__ == "__main__":
    tool = BookTestDriveTool()
    
    # 测试1: 预约E5 Sportback试驾（Mock数据中已有）
    result = tool.run({
        "car_series": "E5 Sportback",
        "dealer": "上海郑隆奥翟",
        "appointment_time": "2026-10-29T14:00:00",
        "customer_name": "张先生",
        "customer_phone": "13800138000"
    })
    print(result.to_json())
    
    # 测试2: 预约新的试驾时间
    result = tool.run({
        "car_series": "E5 Sportback",
        "dealer": "上海郑隆奥翟",
        "appointment_time": "2026-11-01T10:00:00",
        "customer_name": "李女士",
        "customer_phone": "13900139000",
        "notes": "希望试驾高速路段"
    })
    print(result.to_json())
    
    # 测试3: 参数错误（手机号格式错误）
    result = tool.run({
        "car_series": "E5 Sportback",
        "dealer": "上海郑隆奥翟",
        "appointment_time": "2026-10-29T14:00:00",
        "customer_name": "王先生",
        "customer_phone": "123"  # 错误的手机号
    })
    print(result.to_json())
