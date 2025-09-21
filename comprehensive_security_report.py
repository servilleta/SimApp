#!/usr/bin/env python3
"""
Comprehensive Security Report Generator
Consolidates all penetration testing results and generates executive summary
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("security_report")

class SecurityReportGenerator:
    """Generate comprehensive security assessment report"""
    
    def __init__(self, project_dir: str = "/home/paperspace/PROJECT"):
        self.project_dir = project_dir
        self.report_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "project_directory": project_dir,
            "executive_summary": {},
            "detailed_findings": {},
            "recommendations": [],
            "risk_assessment": {},
            "compliance_status": {}
        }
    
    def load_test_results(self):
        """Load all penetration test results"""
        test_files = [
            "network_scan_results.json",
            "sql_injection_results.json", 
            "xss_csrf_results.json",
            "pen_test_results.json"
        ]
        
        results = {}
        
        for test_file in test_files:
            file_path = os.path.join(self.project_dir, test_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        results[test_file.replace('.json', '')] = json.load(f)
                    logger.info(f"Loaded {test_file}")
                except Exception as e:
                    logger.error(f"Failed to load {test_file}: {e}")
            else:
                logger.warning(f"Test results file not found: {test_file}")
        
        return results
    
    def analyze_architecture_security(self) -> Dict[str, Any]:
        """Analyze platform architecture for security issues"""
        # Read docker-compose for architecture analysis
        docker_compose_path = os.path.join(self.project_dir, "docker-compose.yml")
        security_issues = []
        
        if os.path.exists(docker_compose_path):
            try:
                with open(docker_compose_path, 'r') as f:
                    docker_content = f.read()
                
                # Check for security issues in Docker configuration
                if "privileged: true" in docker_content:
                    security_issues.append({
                        "type": "privileged_containers",
                        "severity": "high",
                        "description": "Privileged containers detected in Docker configuration"
                    })
                
                if "network_mode: host" in docker_content:
                    security_issues.append({
                        "type": "host_networking",
                        "severity": "medium", 
                        "description": "Host networking mode bypasses Docker network isolation"
                    })
                
                # Check for exposed secrets
                if "password" in docker_content.lower():
                    security_issues.append({
                        "type": "hardcoded_secrets",
                        "severity": "high",
                        "description": "Potential hardcoded passwords in Docker configuration"
                    })
                
            except Exception as e:
                logger.error(f"Failed to analyze Docker configuration: {e}")
        
        return {
            "issues_found": len(security_issues),
            "security_issues": security_issues
        }
    
    def calculate_risk_score(self, test_results: Dict[str, Any]) -> int:
        """Calculate overall security risk score (0-100, higher is worse)"""
        risk_score = 0
        
        # Network scan results
        if "network_scan_results" in test_results:
            network_data = test_results["network_scan_results"]
            if "summary" in network_data:
                summary = network_data["summary"]
                risk_score += summary.get("critical_findings", 0) * 15
                risk_score += summary.get("high_findings", 0) * 10
                risk_score += summary.get("medium_findings", 0) * 3
                risk_score += summary.get("low_findings", 0) * 1
        
        # SQL injection results
        if "sql_injection_results" in test_results:
            sql_data = test_results["sql_injection_results"]
            if "summary" in sql_data:
                sql_vulns = sql_data["summary"].get("total_vulnerabilities", 0)
                risk_score += sql_vulns * 20  # SQL injection is critical
        
        # XSS/CSRF results
        if "xss_csrf_results" in test_results:
            xss_data = test_results["xss_csrf_results"]
            if "summary" in xss_data:
                summary = xss_data["summary"]
                risk_score += summary.get("total_xss_vulnerabilities", 0) * 8
                risk_score += summary.get("total_csrf_vulnerabilities", 0) * 6
        
        # Cap at 100
        return min(risk_score, 100)
    
    def categorize_risk_level(self, risk_score: int) -> str:
        """Categorize risk level based on score"""
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        elif risk_score >= 20:
            return "LOW"
        else:
            return "MINIMAL"
    
    def generate_executive_summary(self, test_results: Dict[str, Any], risk_score: int) -> Dict[str, Any]:
        """Generate executive summary for leadership"""
        
        total_vulnerabilities = 0
        critical_vulnerabilities = 0
        high_vulnerabilities = 0
        
        # Count vulnerabilities across all tests
        for test_name, data in test_results.items():
            if "summary" in data:
                summary = data["summary"]
                if "total_vulnerabilities" in summary:
                    total_vulnerabilities += summary["total_vulnerabilities"]
                if "critical_vulnerabilities" in summary:
                    critical_vulnerabilities += summary["critical_vulnerabilities"]
                if "high_vulnerabilities" in summary:
                    high_vulnerabilities += summary["high_vulnerabilities"]
                if "total_xss_vulnerabilities" in summary:
                    total_vulnerabilities += summary["total_xss_vulnerabilities"]
                if "total_csrf_vulnerabilities" in summary:
                    total_vulnerabilities += summary["total_csrf_vulnerabilities"]
        
        risk_level = self.categorize_risk_level(risk_score)
        
        # Generate recommendations based on findings
        immediate_actions = []
        strategic_actions = []
        
        if critical_vulnerabilities > 0:
            immediate_actions.append("Address critical vulnerabilities immediately")
            immediate_actions.append("Consider taking system offline until critical issues are resolved")
        
        if high_vulnerabilities > 0:
            immediate_actions.append("Prioritize high-severity vulnerability fixes")
            strategic_actions.append("Implement comprehensive security testing in CI/CD pipeline")
        
        if risk_score > 40:
            strategic_actions.extend([
                "Engage external security consultants for comprehensive audit",
                "Implement Web Application Firewall (WAF)",
                "Establish security monitoring and incident response procedures"
            ])
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "total_vulnerabilities": total_vulnerabilities,
            "critical_vulnerabilities": critical_vulnerabilities,
            "high_vulnerabilities": high_vulnerabilities,
            "immediate_actions": immediate_actions,
            "strategic_actions": strategic_actions,
            "assessment_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "next_assessment_due": "30 days for critical, 90 days for medium risk"
        }
    
    def generate_compliance_status(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compliance status for common frameworks"""
        
        compliance_issues = {
            "OWASP_Top_10": [],
            "NIST_Cybersecurity": [],
            "ISO_27001": [],
            "SOC_2": []
        }
        
        # Check OWASP Top 10 compliance
        if "sql_injection_results" in test_results:
            sql_vulns = test_results["sql_injection_results"]["summary"].get("total_vulnerabilities", 0)
            if sql_vulns > 0:
                compliance_issues["OWASP_Top_10"].append("A03:2021 - Injection vulnerabilities found")
        
        if "xss_csrf_results" in test_results:
            xss_vulns = test_results["xss_csrf_results"]["summary"].get("total_xss_vulnerabilities", 0)
            if xss_vulns > 0:
                compliance_issues["OWASP_Top_10"].append("A07:2021 - Cross-Site Scripting vulnerabilities found")
            
            csrf_vulns = test_results["xss_csrf_results"]["summary"].get("total_csrf_vulnerabilities", 0)
            if csrf_vulns > 0:
                compliance_issues["OWASP_Top_10"].append("A01:2021 - Broken Access Control (CSRF) found")
        
        # NIST Cybersecurity Framework
        if any(compliance_issues["OWASP_Top_10"]):
            compliance_issues["NIST_Cybersecurity"].append("ID.RA - Risk assessment shows vulnerabilities")
            compliance_issues["NIST_Cybersecurity"].append("PR.DS - Data security controls need improvement")
        
        # Calculate compliance scores
        compliance_scores = {}
        for framework, issues in compliance_issues.items():
            if not issues:
                compliance_scores[framework] = "COMPLIANT"
            elif len(issues) <= 2:
                compliance_scores[framework] = "PARTIALLY_COMPLIANT"
            else:
                compliance_scores[framework] = "NON_COMPLIANT"
        
        return {
            "framework_status": compliance_scores,
            "identified_issues": compliance_issues,
            "recommendations": [
                "Implement OWASP secure coding practices",
                "Establish NIST Cybersecurity Framework implementation plan",
                "Conduct ISO 27001 risk assessment",
                "Review SOC 2 Type II audit requirements"
            ]
        }
    
    def generate_detailed_recommendations(self, test_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed security recommendations"""
        
        recommendations = []
        
        # High-priority recommendations based on findings
        if "xss_csrf_results" in test_results:
            xss_data = test_results["xss_csrf_results"]
            if xss_data["summary"]["total_xss_vulnerabilities"] > 0:
                recommendations.append({
                    "priority": "HIGH",
                    "category": "Input Validation",
                    "title": "Implement Comprehensive XSS Protection",
                    "description": "XSS vulnerabilities detected in multiple contexts",
                    "actions": [
                        "Implement Content Security Policy (CSP) with strict directives",
                        "Use proper output encoding for all user input",
                        "Implement input validation with allowlists",
                        "Use template engines with automatic escaping"
                    ],
                    "timeline": "Immediate (1-2 weeks)"
                })
        
        if "network_scan_results" in test_results:
            network_data = test_results["network_scan_results"]
            if network_data["summary"]["high_findings"] > 0:
                recommendations.append({
                    "priority": "HIGH", 
                    "category": "Infrastructure Security",
                    "title": "Secure Docker Container Configuration",
                    "description": "Multiple container security issues identified",
                    "actions": [
                        "Remove privileged container configurations",
                        "Implement non-root container users",
                        "Restrict volume mounts to necessary paths only",
                        "Implement container security scanning in CI/CD"
                    ],
                    "timeline": "1-2 weeks"
                })
        
        # Medium-priority recommendations
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Security Monitoring",
            "title": "Implement Security Monitoring and SIEM",
            "description": "Establish continuous security monitoring capabilities",
            "actions": [
                "Deploy Web Application Firewall (WAF)",
                "Implement security event logging and monitoring",
                "Set up automated vulnerability scanning",
                "Establish incident response procedures"
            ],
            "timeline": "4-6 weeks"
        })
        
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Access Control",
            "title": "Enhance Authentication and Authorization",
            "description": "Strengthen authentication mechanisms",
            "actions": [
                "Implement multi-factor authentication (MFA)",
                "Establish role-based access control (RBAC)",
                "Implement session management best practices",
                "Regular access reviews and deprovisioning"
            ],
            "timeline": "3-4 weeks"
        })
        
        # Low-priority strategic recommendations
        recommendations.append({
            "priority": "LOW",
            "category": "Security Culture",
            "title": "Establish Security Development Lifecycle",
            "description": "Build security into development processes",
            "actions": [
                "Security training for development team",
                "Implement secure code review processes",
                "Regular penetration testing schedule",
                "Security champions program"
            ],
            "timeline": "8-12 weeks"
        })
        
        return recommendations
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate the complete security assessment report"""
        logger.info("Generating comprehensive security report...")
        
        # Load all test results
        test_results = self.load_test_results()
        
        # Analyze architecture
        architecture_analysis = self.analyze_architecture_security()
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(test_results)
        
        # Generate executive summary
        executive_summary = self.generate_executive_summary(test_results, risk_score)
        
        # Generate compliance status
        compliance_status = self.generate_compliance_status(test_results)
        
        # Generate detailed recommendations
        recommendations = self.generate_detailed_recommendations(test_results)
        
        # Compile final report
        self.report_data.update({
            "executive_summary": executive_summary,
            "detailed_findings": test_results,
            "architecture_analysis": architecture_analysis,
            "recommendations": recommendations,
            "risk_assessment": {
                "overall_score": risk_score,
                "risk_level": self.categorize_risk_level(risk_score),
                "calculation_methodology": "Weighted scoring based on vulnerability severity and count"
            },
            "compliance_status": compliance_status
        })
        
        return self.report_data
    
    def save_report(self, filename: str = "comprehensive_security_assessment_report.json"):
        """Save the comprehensive report to file"""
        report_path = os.path.join(self.project_dir, filename)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(self.report_data, f, indent=2)
            logger.info(f"Comprehensive security report saved to: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return None
    
    def print_executive_summary(self):
        """Print executive summary to console"""
        summary = self.report_data["executive_summary"]
        
        print(f"\n{'='*80}")
        print(f"MONTE CARLO PLATFORM - SECURITY ASSESSMENT EXECUTIVE SUMMARY")
        print(f"{'='*80}")
        print(f"Assessment Date: {summary['assessment_date']}")
        print(f"Overall Risk Score: {summary['risk_score']}/100 ({summary['risk_level']})")
        print(f"Total Vulnerabilities: {summary['total_vulnerabilities']}")
        print(f"Critical Vulnerabilities: {summary['critical_vulnerabilities']}")
        print(f"High Vulnerabilities: {summary['high_vulnerabilities']}")
        
        if summary['immediate_actions']:
            print(f"\n🚨 IMMEDIATE ACTIONS REQUIRED:")
            for action in summary['immediate_actions']:
                print(f"   • {action}")
        
        if summary['strategic_actions']:
            print(f"\n📋 STRATEGIC RECOMMENDATIONS:")
            for action in summary['strategic_actions']:
                print(f"   • {action}")
        
        print(f"\nNext Assessment Due: {summary['next_assessment_due']}")
        print(f"{'='*80}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Security Report Generator")
    parser.add_argument("--project-dir", default="/home/paperspace/PROJECT", help="Project directory")
    parser.add_argument("--output", default="comprehensive_security_assessment_report.json", help="Output file")
    
    args = parser.parse_args()
    
    # Generate comprehensive report
    generator = SecurityReportGenerator(args.project_dir)
    report = generator.generate_comprehensive_report()
    
    # Save report
    report_path = generator.save_report(args.output)
    
    # Print executive summary
    generator.print_executive_summary()
    
    # Return appropriate exit code
    risk_level = report["executive_summary"]["risk_level"]
    if risk_level in ["CRITICAL", "HIGH"]:
        return 1
    else:
        return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
