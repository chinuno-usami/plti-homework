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

    use crate::{nameless, stack_machine_named};
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
    type CompileEnv = LinkedList<String>;
    pub fn compile_to_nameless(expr: Box<Expr>) -> Box<nameless::Expr> {
        compile_to_nameless_impl(expr, [].into())
    }
    fn compile_to_nameless_impl(expr: Box<Expr>, env: CompileEnv) -> Box<nameless::Expr> {
        match *expr {
            Expr::Cst(val) => Box::new(nameless::Expr::Cst(val)),
            Expr::Add(expr1, expr2) => Box::new(nameless::Expr::Add(compile_to_nameless_impl(expr1, env.clone()), compile_to_nameless_impl(expr2, env.clone()))),
            Expr::Mul(expr1, expr2) => Box::new(nameless::Expr::Mul(compile_to_nameless_impl(expr1, env.clone()), compile_to_nameless_impl(expr2, env.clone()))),
            Expr::Var(name) => Box::new(nameless::Expr::Var(env.iter().position(|it| *it == name).unwrap())),
            Expr::Let(name, expr1, expr2) => {
                let new_env: CompileEnv = [name].into_iter().chain(env.into_iter()).collect();
                Box::new(nameless::Expr::Let(compile_to_nameless_impl(expr1, new_env.clone()), compile_to_nameless_impl(expr2, new_env.clone())))
            }
        }
    }
    
    pub fn compile_to_named(expr: Box<Expr>) -> stack_machine_named::Instrs {
        match *expr {
            Expr::Cst(val) => [stack_machine_named::Instr::Cst(val)].into(),
            Expr::Add(expr1, expr2) => {
                let tmp2 = compile_to_named(expr2);
                let tmp1 = compile_to_named(expr1);
                tmp1.into_iter()
                    .chain(tmp2.into_iter())
                    .chain([stack_machine_named::Instr::Add].into_iter()).collect()
            }
            Expr::Mul(expr1, expr2) => {
                let tmp2 = compile_to_named(expr2);
                let tmp1 = compile_to_named(expr1);
                tmp1.into_iter()
                    .chain(tmp2.into_iter())
                    .chain([stack_machine_named::Instr::Mul].into_iter()).collect()
            }
            Expr::Var(name) => [stack_machine_named::Instr::Var(name)].into(),
            Expr::Let(name, expr1, expr2) => {
                let tmp1 = compile_to_named(expr1);
                let tmp2 = compile_to_named(expr2);
                tmp1.into_iter()
                    .chain([stack_machine_named::Instr::Let(name)]).into_iter()
                    .chain(tmp2.into_iter())
                    .chain(
                [stack_machine_named::Instr::LetEnd].into_iter()).collect()
            }
        }
    }
}

pub mod interpreter_with_name {
    use crate::ast;
    use std::collections::LinkedList;

    type Env = LinkedList<(String, i32)>;
    pub fn eval(expr: Box<ast::Expr>) -> i32 {
        let env = Env::new();
        eval_impl(expr, env)
    }
    fn eval_impl(expr: Box<ast::Expr>, env: Env) -> i32 {
        match *expr {
            ast::Expr::Cst(val) => val,
            ast::Expr::Add(expr1, expr2) => eval_impl(expr1, env.clone()) + eval_impl(expr2, env.clone()),
            ast::Expr::Mul(expr1, expr2) => eval_impl(expr1, env.clone()) * eval_impl(expr2, env.clone()),
            ast::Expr::Var(name) => env.iter().find(|&it| it.0 == name).unwrap().1,
            ast::Expr::Let(name, expr1, expr2) => {
                let sub_result = eval_impl(expr1, env.clone());
                let env = [(name.clone(), sub_result)]
                    .into_iter()
                    .chain(env.into_iter())
                    .collect();
                eval_impl(expr2, env)
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

    type Env = LinkedList<i32>;
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

// 虚线路径：带名称的本地变量+临时变量放栈里作为env
// 不是很理解，猜测和最终栈机指令一样，但是本地变量还是带名称形式，临时变量先行实现放栈里
pub mod stack_machine_named{
    use std::collections::LinkedList;

    use crate::stack_machine;

    type Operand = i32;
    //use i32 as Operand;
    #[derive(Clone, Debug)]
    pub enum Instr {
        Cst(Operand),
        Add,
        Mul,
        Var(String), // 取址
        Let(String), // 本地变量
        LetEnd, // 临时变量作用域结束
    }

    pub type Instrs = LinkedList<Instr>;
    type Stack = LinkedList<Operand>;
    type LocalVar = LinkedList<(String, Operand)>;

    #[derive(Debug)]
    struct Env {
        stack: Stack,
        local: LocalVar,
    }
    
    impl Env {
        pub fn new() -> Env {
            Env{
                stack: Stack::new(),
                local: LocalVar::new()
            }
        }
    }
    
    pub fn eval(instrs: Instrs) -> Operand{
        let mut env = Env::new();
        eval_impl(instrs, &mut env)
    }
    fn eval_impl(mut instrs: Instrs, env:&mut Env) -> Operand{
        while let Some(instr) = instrs.pop_front() {
            // debug
            // println!("instrs:{:?}:{:?}, env:{:?}", instr, instrs, env);
            match instr {
                Instr::Cst(val) => env.stack.push_front(val),
                Instr::Add => {
                    let op1 = env.stack.pop_front().unwrap();
                    let op2 = env.stack.pop_front().unwrap();
                    env.stack.push_front(op1+op2)
                },
                Instr::Mul => {
                    let op1 = env.stack.pop_front().unwrap();
                    let op2 = env.stack.pop_front().unwrap();
                    env.stack.push_front(op1*op2)
                },
                Instr::Var(name) => {
                    env.stack.push_front(env.local.iter().find(|it| it.0 == name).unwrap().1)
                },
                Instr::Let(name) => {
                    let val = env.stack.pop_front().unwrap();
                    env.local.push_front((name, val))
                },
                Instr::LetEnd => {
                    env.local.pop_front();
                }
            }
        }
        env.stack.pop_front().unwrap()
    }

    // Some(String) => local
    // None => temp
    type CompileEnv = LinkedList<Option<String>>;

    pub fn compile(instrs: Instrs) -> stack_machine::Instrs {
        let mut env = CompileEnv::new();
        compile_impl(instrs, &mut env)
    }
    pub fn compile_impl(mut instrs: Instrs, env:&mut CompileEnv) -> stack_machine::Instrs {
        let mut output = stack_machine::Instrs::new();
        while let Some(instr) = instrs.pop_front() {
            // debug
            // println!("instrs:{:?}:{:?}, env:{:?}", instr, instrs, env);
            match instr {
                Instr::Cst(val) => {
                    output.push_back(stack_machine::Instr::Cst(val));
                    env.push_front(None);
                },
                Instr::Add => {
                    output.push_back(stack_machine::Instr::Add);
                    env.pop_front();
                },
                Instr::Mul => {
                    output.push_back(stack_machine::Instr::Mul);
                    env.pop_front();
                },
                Instr::Var(name) => {
                    let idx = {
                        env.iter().position(|it| {
                            match it {
                                Some(var) => {
                                    *var == name
                                },
                                None => false
                            }
                        })
                    };
                    output.push_back(stack_machine::Instr::Var(idx.unwrap()));
                    env.push_front(None);
                },
                Instr::Let(name) => {
                    env.pop_front();
                    env.push_front(Some(name));
                },
                Instr::LetEnd => {
                    env.pop_front();
                    output.push_back(stack_machine::Instr::Swap);
                    output.push_back(stack_machine::Instr::Pop);
                }
            }
        };
        output
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
    type Stack = LinkedList<i32>;

    // interpreter for the stack machine with variables
    pub fn eval(instrs: Instrs) -> Operand {
        let mut stack = Stack::new();
        eval_impl(instrs, &mut stack)
    }
    fn eval_impl(mut instrs: Instrs, stk: &mut Stack) -> Operand {
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
            Some(Instr::Cst(val)) => eval_impl(instrs, {
                stk.push_front(val);
                stk
            }),
            Some(Instr::Add) => eval_impl(instrs, {
                do_op(stk, |a, b| a + b);
                stk
            }),
            Some(Instr::Mul) => eval_impl(instrs, {
                do_op(stk, |a, b| a * b);
                stk
            }),
            Some(Instr::Var(idx)) => {
                stk.push_front(*stk.iter().nth(idx).unwrap());
                eval_impl(instrs, stk)
            }
            Some(Instr::Pop) => {
                stk.pop_front();
                eval_impl(instrs, stk)
            }
            Some(Instr::Swap) => {
                let top = stk.pop_front().unwrap();
                let second = stk.pop_front().unwrap();
                stk.push_front(top);
                stk.push_front(second);
                eval_impl(instrs, stk)
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
    
    // let ( x = 2 in let ( x = x +1 ) in x * 3 end )
    let test_expr2 = r#let!(
        "x".into(),
        cst!(2),
        r#let!(
            "x".into(),
            add!(
                var!("x".into()),
                cst!(1)
            ),
            mul!(
                var!("x".into()),
                cst!(3)
            )
        )
    );
    println!("test_expr:{:?}", test_expr);
    println!("test_expr2:{:?}", test_expr2);
    println!("expr interpreter result:{}", interpreter_with_name::eval(test_expr.clone()));
    println!("expr2 interpreter result:{}", interpreter_with_name::eval(test_expr2.clone()));
    let nameless_expr = ast::compile_to_nameless(test_expr.clone());
    let nameless_expr2 = ast::compile_to_nameless(test_expr2.clone());
    println!("expr nameless_expr:{:?}", nameless_expr);
    println!("expr2 nameless_expr:{:?}", nameless_expr2);
    println!("expr nameless result:{}", nameless::eval(nameless_expr.clone(), [].into()));
    println!("expr2 nameless result:{}", nameless::eval(nameless_expr2.clone(), [].into()));
    let stack_instrs = nameless::compile(nameless_expr);
    let stack_instrs2 = nameless::compile(nameless_expr2);
    println!("expr stack_instrs:{:?}",stack_instrs.clone());
    println!("expr2 stack_instrs:{:?}",stack_instrs2.clone());
    println!("expr stack machine result:{}",stack_machine::eval(stack_instrs));
    println!("expr2 stack machine result:{}",stack_machine::eval(stack_instrs2));
    let named_stack_instrs = ast::compile_to_named(test_expr.clone());
    let named_stack_instrs2 = ast::compile_to_named(test_expr2.clone());
    println!("expr named stack_instrs:{:?}",named_stack_instrs.clone());
    println!("expr2 named stack_instrs:{:?}",named_stack_instrs2.clone());
    println!("expr named interpreter result:{}", stack_machine_named::eval(named_stack_instrs.clone()));
    println!("expr2 named interpreter result:{}", stack_machine_named::eval(named_stack_instrs2.clone()));
    let named_stack_instrs = stack_machine_named::compile(named_stack_instrs);
    let named_stack_instrs2 = stack_machine_named::compile(named_stack_instrs2);
    println!("expr named stack machine result:{}", stack_machine::eval(named_stack_instrs));
    println!("expr2 named stack machine result:{}", stack_machine::eval(named_stack_instrs2));

}
