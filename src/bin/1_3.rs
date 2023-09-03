/// concrete syntax BNF
/// expr : Int
///     | expr "+" expr
///     | expr "*" expr
///     | "(" expr ")"
///     | expr "=" expr
///     | "let" expr "in" expr "end"

// abstract syntax
pub mod ast {
    use std::collections::LinkedList;

    use crate::nameless;
    // Expr编译期需要知道具体大小，递归定义只能用指针存储
    #[derive(Clone, Debug)]
    pub enum Expr {
        Cst(i32),
        Add(Box<Expr>, Box<Expr>),
        Mul(Box<Expr>, Box<Expr>),
        Var(String),
        Let(String, Box<Expr>, Box<Expr>),
    }

    // 名字和序号关联
    pub type CompileEnv = LinkedList<String>;
    pub fn compile_to_nameless(expr: Box<Expr>, env: CompileEnv) -> Box<nameless::Expr> {
        match *expr {
            Expr::Cst(val) => Box::new(nameless::Expr::Cst(val)),
            Expr::Add(expr1, expr2) => Box::new(nameless::Expr::Add(compile_to_nameless(expr1, env.clone()), compile_to_nameless(expr2, env.clone()))),
            Expr::Mul(expr1, expr2) => Box::new(nameless::Expr::Mul(compile_to_nameless(expr1, env.clone()), compile_to_nameless(expr2, env.clone()))),
            Expr::Var(name) => Box::new(nameless::Expr::Var(env.iter().position(|it| *it == name).unwrap())),
            Expr::Let(name, expr1, expr2) => {
                let new_env: CompileEnv = [name].into_iter().chain(env.into_iter()).collect();
                Box::new(nameless::Expr::Let(compile_to_nameless(expr1, new_env.clone()), compile_to_nameless(expr2, new_env.clone())))
            }
        }
    }
}

pub mod interpreter_with_name {
    use crate::ast;
    use std::collections::LinkedList;

    pub type Env = LinkedList<(String, i32)>;
    pub fn eval(expr: Box<ast::Expr>, env: Env) -> i32 {
        match *expr {
            ast::Expr::Cst(val) => val,
            ast::Expr::Add(expr1, expr2) => eval(expr1, env.clone()) + eval(expr2, env.clone()),
            ast::Expr::Mul(expr1, expr2) => eval(expr1, env.clone()) * eval(expr2, env.clone()),
            ast::Expr::Var(name) => env.iter().find(|&it| it.0 == name).unwrap().1,
            ast::Expr::Let(name, expr1, expr2) => {
                let sub_result = eval(expr1, env.clone());
                let env = [(name.clone(), sub_result)]
                    .into_iter()
                    .chain(env.into_iter())
                    .collect();
                eval(expr2, env)
            }
        }
    }
}

pub mod nameless {
    use std::collections::LinkedList;
    use crate::stack_machine;

    #[derive(Clone, Debug)]
    pub enum Expr {
        Cst(i32),
        Add(Box<Expr>, Box<Expr>),
        Mul(Box<Expr>, Box<Expr>),
        Var(usize),
        Let(Box<Expr>, Box<Expr>),
    }

    pub type Env = LinkedList<i32>;
    pub fn eval(expr: Box<Expr>, env: Env) -> i32 {
        match *expr {
            Expr::Cst(val) => val,
            Expr::Add(expr1, expr2) => eval(expr1, env.clone()) + eval(expr2, env.clone()),
            Expr::Mul(expr1, expr2) => eval(expr1, env.clone()) * eval(expr2, env.clone()),
            Expr::Var(idx) => *env.iter().nth(idx).unwrap(),
            Expr::Let(expr1, expr2) => {
                let sub_result = eval(expr1, env.clone());
                let env = [sub_result].into_iter().chain(env.into_iter()).collect();
                eval(expr2, env)
            }
        }
    }
    
    pub fn compile(expr: Box<Expr>) -> stack_machine::Instrs {
        compile_impl(expr, 0)
    }
    fn compile_impl(expr: Box<Expr>, offset: usize) -> stack_machine::Instrs {
        match *expr {
            Expr::Cst(i) => {
                stack_machine::Instrs::from([stack_machine::Instr::Cst(i)])
            },
            Expr::Add(expr1, expr2) => {
                let mut stack1 = compile_impl(expr1, offset);
                // 对于二元操作，expr1结果入栈，所以计算expr2的时候取址偏移+1
                let mut stack2 = compile_impl(expr2, offset+1);
                stack1.append(&mut stack2);
                stack1.push_back(stack_machine::Instr::Add);
                stack1
            },
            Expr::Mul(expr1, expr2) => {
                let mut stack1 = compile_impl(expr1, offset);
                let mut stack2 = compile_impl(expr2, offset+1);
                stack1.append(&mut stack2);
                stack1.push_back(stack_machine::Instr::Mul);
                stack1
            }
            Expr::Var(idx) => {
                stack_machine::Instrs::from([stack_machine::Instr::Var(idx+offset)])
            },
            Expr::Let(expr1, expr2) => {
                let mut stack1 = compile_impl(expr1, offset);
                let mut stack2 = compile_impl(expr2, offset);
                let mut balance = stack_machine::Instrs::from([stack_machine::Instr::Swap, stack_machine::Instr::Pop]);
                stack1.append(&mut stack2);
                stack1.append(&mut balance);
                stack1
            }
        }
    }
}

pub mod stack_machine {
    use std::collections::LinkedList;

    type Operand = i32;
    //use i32 as Operand;
    #[derive(Clone, Debug)]
    pub enum Instr {
        Cst(Operand),
        Add,
        Mul,
        Var(usize),
        Pop,
        Swap,
    }

    pub type Instrs = LinkedList<Instr>;
    pub type Stack = LinkedList<i32>;

    // interpreter for the stack machine with variables
    pub fn eval(mut instrs: Instrs, stk: &mut Stack) -> Operand {
        // debug step
        // println!("instr:{:?}, stk:{:?}", instrs, stk);
        // linkedList的实现不好进行模式匹配，添加一个辅助函数来简化操作
        let do_op = |stk: &mut Stack, op: fn(Operand, Operand) -> Operand| {
            let a = stk.pop_front();
            let b = stk.pop_front();
            match (a, b) {
                (Some(a1), Some(b1)) => {
                    stk.push_front(op(a1, b1));
                }
                _ => panic!(),
            }
        };
        match instrs.pop_front() {
            Some(Instr::Cst(val)) => eval(instrs, {
                stk.push_front(val);
                stk
            }),
            Some(Instr::Add) => eval(instrs, {
                do_op(stk, |a, b| a + b);
                stk
            }),
            Some(Instr::Mul) => eval(instrs, {
                do_op(stk, |a, b| a * b);
                stk
            }),
            Some(Instr::Var(idx)) => {
                stk.push_front(*stk.iter().nth(idx).unwrap());
                eval(instrs, stk)
            }
            Some(Instr::Pop) => {
                stk.pop_front();
                eval(instrs, stk)
            }
            Some(Instr::Swap) => {
                let top = stk.pop_front().unwrap();
                let second = stk.pop_front().unwrap();
                stk.push_front(top);
                stk.push_front(second);
                eval(instrs, stk)
            }
            None => *stk.front().unwrap(),
        }
    }
}

// 简化new操作的宏
macro_rules! cst {
    ($e: expr) => {
        Box::new(ast::Expr::Cst($e))
    };
}

macro_rules! add {
    ($($e: expr),*) => {
        Box::new(ast::Expr::Add($($e,)*))
    };
}

macro_rules! mul {
    ($($e: expr),*) => {
        Box::new(ast::Expr::Mul($($e,)*))
    };
}

macro_rules! var {
    ($e: expr) => {
        Box::new(ast::Expr::Var($e))
    };
}

macro_rules! r#let {
    ($($e: expr),*) => {
        Box::new(ast::Expr::Let($($e,)*))
    };
}

fn main() {
    // test interpreter
    // let ( x=2 in let (y=3 in y + ( x * 4 ) end ) * (x + 1) end )
    let test_expr = r#let!(
        "x".into(),
        cst!(2),
        mul!(
            r#let!(
                "y".into(), 
                cst!(3),
                add!(
                    var!("y".into()),
                    mul!(
                        var!("x".into()),
                        cst!(4)
                    )
                )
            ),
            add!(
                var!("x".into()),
                cst!(1)
            )
        )
    );
    println!("test_expr:{:?}", test_expr);
    println!("interpreter result:{}", interpreter_with_name::eval(test_expr.clone(), [].into()));
    let nameless_expr = ast::compile_to_nameless(test_expr.clone(), [].into());
    println!("nameless_expr:{:?}", nameless_expr);
    println!("nameless result:{}", nameless::eval(nameless_expr.clone(), [].into()));
    let stack_instrs = nameless::compile(nameless_expr);
    println!("stack_instrs:{:?}",stack_instrs.clone());
    let mut stack = stack_machine::Stack::new();
    println!("stack machine result:{}",stack_machine::eval(stack_instrs, &mut stack));

}
